#!/usr/bin/env python
# coding=utf-8
"""
Bounded, proportional dataset downloader for DivMoE Stage 1/2 pre-training data.

Writes data into the directory layout expected by prepare_data/olmoe_dataset.py:

    <out_dir>/<source>/<...>.jsonl.gz   (or the original shard names in files mode)

Only the requested per-source token budget is downloaded, so the full corpus is
never fetched.

Two access modes:
  * configs: each source is a HuggingFace *dataset config* (e.g. Dolmino-Mix).
  * files:   the repo has a single `default` config and stores subsets as
             subdirectories / raw shard files (e.g. OLMoE-mix-0924). Raw shards
             are downloaded as-is (already in loader format), no re-encoding.
  * auto (default): pick `configs` if the requested source names match config
             names, otherwise `files`.

Token counts: in configs mode estimated from UTF-8 bytes (cheap); in files mode
from the compressed shard size (approximate, conservative). Pass --tokenizer in
configs mode for exact counts (slow).

Examples:
    # Stage 1, one command per domain (OLMoE-mix-0924, auto -> files mode)
    python -m prepare_data.download_subset --out-dir data/stage1/math     --preset stage1_math
    python -m prepare_data.download_subset --out-dir data/stage1/code     --preset stage1_code
    python -m prepare_data.download_subset --out-dir data/stage1/academic --preset stage1_academic
    python -m prepare_data.download_subset --out-dir data/stage1/web      --preset stage1_web

    # Stage 2 (500B tokens total, capped per source)
    python -m prepare_data.download_subset --out-dir data/stage2 --preset stage2

    # Quick smoke test: 1% of every budget
    python -m prepare_data.download_subset --out-dir /tmp/mini --preset stage2 --scale 0.01

    # Paper text corpus (Dolmino-Mix, named configs -> configs mode)
    python -m prepare_data.download_subset --repo allenai/dolmino-mix-1124 \
        --out-dir data/stage2 --preset stage2 \
        --config-map '{"open-web-math":"math","algebraic-stack":"math"}'

    # Inspect what a repo offers before downloading
    python -m prepare_data.download_subset --repo allenai/OLMoE-mix-0924 --inspect
"""

import argparse
import gzip
import json
import os
import sys
from typing import Dict, List

G = 1_000_000_000

# Budgets are deliberately conservative; small sources stop early when exhausted
# and are up-sampled by the training loader (which cycles over files by weight).
PRESETS: Dict[str, Dict[str, int]] = {
    "stage1_math": {"open-web-math": 5 * G, "algebraic-stack": 5 * G},
    "stage1_code": {"starcoder": 10 * G},
    "stage1_academic": {"pes2o": 10 * G},
    "stage1_web": {"dclm": 6 * G, "wiki": int(3.7 * G)},
    "stage2": {
        "dclm": 200 * G,
        "starcoder": 75 * G,
        "pes2o": 75 * G,
        "algebraic-stack": 11 * G,
        "open-web-math": 14 * G,
        "wiki": int(3.7 * G),
    },
}

DEFAULT_REPO = "allenai/OLMoE-mix-0924"
JSON_SUFFIXES = (".json", ".jsonl", ".json.gz", ".jsonl.gz", ".json.zst", ".jsonl.zst")


def _norm(name: str) -> str:
    return name.lower().replace("_", "-")


def _is_json(path: str) -> bool:
    return path.endswith(JSON_SUFFIXES)


def count_tokens(text: str, tokenizer, bytes_per_token: float) -> int:
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    return max(1, int(len(text.encode("utf-8")) / bytes_per_token))


# --------------------------------------------------------------------------- #
# configs mode
# --------------------------------------------------------------------------- #
class ShardWriter:
    """Write JSONL records into rolling gzip shards of ~shard_tokens each."""

    def __init__(self, out_dir: str, source: str, shard_tokens: int):
        self.dir = os.path.join(out_dir, source)
        os.makedirs(self.dir, exist_ok=True)
        self.source = source
        self.shard_tokens = max(1, shard_tokens)
        self.idx = 0
        self.in_shard = 0
        self.f = None
        self._open()

    def _open(self):
        self.path = os.path.join(self.dir, f"{self.source}-part-{self.idx:05d}.jsonl.gz")
        self.f = gzip.open(self.path, "wt", encoding="utf-8")

    def write(self, text: str, n_tokens: int, example_id: str = ""):
        if self.in_shard > 0 and self.in_shard + n_tokens > self.shard_tokens:
            self.f.close()
            self.idx += 1
            self.in_shard = 0
            self._open()
        self.f.write(json.dumps({"text": text, "id": str(example_id)}, ensure_ascii=False) + "\n")
        self.in_shard += n_tokens

    def close(self):
        if self.f is not None:
            self.f.close()
            self.f = None


def download_config(repo, source, dataset_config, budget, out_dir, tokenizer,
                    bytes_per_token, shard_tokens):
    from datasets import load_dataset

    print(f"[{source}] streaming {repo}:{dataset_config} (budget {budget / G:.2f}B tokens)",
          file=sys.stderr)
    writer = ShardWriter(out_dir, source, shard_tokens)
    ds = load_dataset(repo, dataset_config, split="train", streaming=True)
    written = 0
    try:
        for example in ds:
            text = example.get("text") or example.get("content") or ""
            if not text:
                continue
            n = count_tokens(text, tokenizer, bytes_per_token)
            writer.write(text, n, example.get("id", ""))
            written += n
            if written >= budget:
                break
    finally:
        writer.close()
    print(f"[{source}] wrote ~{written / G:.2f}B tokens -> {writer.dir}", file=sys.stderr)
    return written


# --------------------------------------------------------------------------- #
# files mode
# --------------------------------------------------------------------------- #
def list_repo_file_sizes(repo: str) -> Dict[str, int]:
    from huggingface_hub import HfApi

    sizes = {}
    for entry in HfApi().list_repo_tree(repo, repo_type="dataset", recursive=True):
        size = getattr(entry, "size", None)
        path = getattr(entry, "path", None)
        if path is not None and size is not None:
            sizes[path] = size
    return sizes


def match_source_files(all_paths: List[str], source: str) -> List[str]:
    target = _norm(source)
    matched = [
        p for p in all_paths
        if _is_json(p) and target in [_norm(part) for part in p.split("/")]
    ]
    return sorted(matched)


def download_files(repo, source, files, budget, out_dir, bytes_per_token):
    from huggingface_hub import hf_hub_download

    if not files:
        print(f"[{source}] no matching files found in {repo}", file=sys.stderr)
        return 0

    sizes = list_repo_file_sizes(repo)
    budget_bytes = int(budget * bytes_per_token)
    picked, used, est_tokens = [], 0, 0
    for path in files:
        size = sizes.get(path, 0)
        if picked and used + size > budget_bytes:
            break
        picked.append(path)
        used += size
        est_tokens += int(size / bytes_per_token)
        if used >= budget_bytes:
            break

    print(f"[{source}] downloading {len(picked)}/{len(files)} shards "
          f"(~{est_tokens / G:.2f}B tokens, {used / G:.2f}GB) -> {out_dir}/{source}",
          file=sys.stderr)
    for path in picked:
        hf_hub_download(repo_id=repo, filename=path, repo_type="dataset", local_dir=out_dir)
    return est_tokens


# --------------------------------------------------------------------------- #
def inspect(repo: str):
    from huggingface_hub import HfApi
    from datasets import get_dataset_config_names

    try:
        cfgs = get_dataset_config_names(repo)
    except Exception as e:  # noqa: BLE001
        cfgs = [f"<unavailable: {e}>"]

    sizes = list_repo_file_sizes(repo)
    print(f"repo: {repo}")
    print(f"configs: {cfgs}")
    print(f"json files: {sum(1 for p in sizes if _is_json(p))} / {len(sizes)} total")

    prefixes: Dict[str, int] = {}
    for path in sizes:
        if not _is_json(path):
            continue
        parts = path.split("/")
        key = parts[0] if len(parts) > 1 else "<root>"
        prefixes[key] = prefixes.get(key, 0) + sizes[path]
    for key, nbytes in sorted(prefixes.items()):
        print(f"  {key}: {nbytes / G:.2f} GB")


def parse_args():
    p = argparse.ArgumentParser(description="Bounded, proportional dataset downloader")
    p.add_argument("--out-dir", help="Directory to write <source>/ data into")
    p.add_argument("--preset", nargs="+", choices=sorted(PRESETS), default=["stage2"],
                   help="One or more budget presets (merged by summing per-source budgets)")
    p.add_argument("--repo", default=DEFAULT_REPO, help="HF dataset repo id")
    p.add_argument("--mode", choices=["auto", "configs", "files"], default="auto")
    p.add_argument("--config-map", default=None,
                   help="JSON mapping of preset source -> dataset config name (configs mode)")
    p.add_argument("--tokenizer", default=None,
                   help="Tokenizer path for exact token counts in configs mode")
    p.add_argument("--bytes-per-token", type=float, default=None,
                   help="configs mode: bytes/token of raw text (~4 English). "
                        "files mode: compressed bytes/token (~1.5). Defaults per mode.")
    p.add_argument("--shard-tokens", type=int, default=500 * 1_000_000,
                   help="Approx tokens per output shard in configs mode")
    p.add_argument("--scale", type=float, default=1.0,
                   help="Scale all budgets, e.g. 0.01 for a quick smoke test")
    p.add_argument("--max-total-tokens", type=float, default=None,
                   help="Global cap across all sources (in tokens)")
    p.add_argument("--inspect", action="store_true",
                   help="Print configs and file layout for --repo, then exit")
    p.add_argument("--list-presets", action="store_true")
    return p.parse_args()


def resolve_mode(args, budgets):
    if args.mode != "auto":
        return args.mode
    try:
        from datasets import get_dataset_config_names
        cfgs = {_norm(c) for c in get_dataset_config_names(args.repo)}
    except Exception:  # noqa: BLE001
        return "configs"
    wanted = {_norm(s) for s in budgets}
    return "configs" if wanted & cfgs else "files"


def main():
    args = parse_args()

    if args.list_presets:
        for name, budgets in PRESETS.items():
            print(f"{name}: {sum(budgets.values()) / G:.1f}B tokens")
            for src, b in budgets.items():
                print(f"    {src}: {b / G:.2f}B")
        return

    if args.inspect:
        inspect(args.repo)
        return

    if not args.out_dir:
        raise SystemExit("error: --out-dir is required (or use --list-presets / --inspect)")

    budgets: Dict[str, int] = {}
    for preset in args.preset:
        for src, b in PRESETS[preset].items():
            budgets[src] = budgets.get(src, 0) + b

    if args.scale != 1.0:
        budgets = {s: int(b * args.scale) for s, b in budgets.items()}

    total_budget = sum(budgets.values())
    if args.max_total_tokens is not None and total_budget > args.max_total_tokens:
        factor = args.max_total_tokens / total_budget
        budgets = {s: int(b * factor) for s, b in budgets.items()}

    mode = resolve_mode(args, budgets)
    bytes_per_token = args.bytes_per_token or (4.0 if mode == "configs" else 1.5)

    print(f"repo={args.repo} mode={mode} presets={args.preset} "
          f"-> {sum(budgets.values()) / G:.2f}B tokens across {len(budgets)} sources",
          file=sys.stderr)

    config_map = json.loads(args.config_map) if args.config_map else {}
    tokenizer = None
    if args.tokenizer and mode == "configs":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    all_paths = None
    if mode == "files":
        all_paths = list(list_repo_file_sizes(args.repo).keys())

    total_written = 0
    for source, budget in budgets.items():
        if mode == "configs":
            total_written += download_config(
                args.repo, source, config_map.get(source, source), budget,
                args.out_dir, tokenizer, bytes_per_token, args.shard_tokens,
            )
        else:
            total_written += download_files(
                args.repo, source, match_source_files(all_paths, source), budget,
                args.out_dir, bytes_per_token,
            )

    print(f"Done. ~{total_written / G:.2f}B tokens into {args.out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
