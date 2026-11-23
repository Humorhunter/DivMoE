#!/usr/bin/env python
# coding=utf-8
"""
Stream a bounded number of tokens from a HuggingFace dataset and write them as
JSONL(.gz) shards in the directory layout expected by prepare_data/olmoe_dataset.py:

    <out_dir>/<source>/<source>-part-00000.jsonl.gz

Only the requested per-source token budget is downloaded, so the full corpus is
never fetched. Token counts are estimated from UTF-8 byte length by default
(cheap); pass --tokenizer for exact counts (much slower).

Examples:
    # Stage 1, one command per domain
    python -m prepare_data.download_subset --out-dir data/stage1/math     --preset stage1_math
    python -m prepare_data.download_subset --out-dir data/stage1/code     --preset stage1_code
    python -m prepare_data.download_subset --out-dir data/stage1/academic --preset stage1_academic
    python -m prepare_data.download_subset --out-dir data/stage1/web      --preset stage1_web

    # Stage 2 (500B tokens total, capped per source)
    python -m prepare_data.download_subset --out-dir data/stage2 --preset stage2

    # Quick smoke test: 1% of every budget
    python -m prepare_data.download_subset --out-dir /tmp/mini --preset stage2 --scale 0.01

    # Paper text corpus instead of the repo loader default
    python -m prepare_data.download_subset --repo allenai/dolmino-mix-1124 \
        --out-dir data/stage2 --preset stage2 \
        --config-map '{"open-web-math":"math","stackexchange":"stackexchange","pes2o":"pes2o","dclm":"dclm","wiki":"wiki","starcoder":"starcoder"}'
"""

import argparse
import gzip
import json
import os
import sys
from typing import Dict

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


def count_tokens(text: str, tokenizer, bytes_per_token: float) -> int:
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    return max(1, int(len(text.encode("utf-8")) / bytes_per_token))


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


def download_source(repo, source, dataset_config, budget, out_dir, tokenizer,
                    bytes_per_token, shard_tokens):
    print(f"[{source}] streaming {repo}:{dataset_config} (budget {budget / G:.2f}B tokens)",
          file=sys.stderr)
    writer = ShardWriter(out_dir, source, shard_tokens)
    from datasets import load_dataset
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


def parse_args():
    p = argparse.ArgumentParser(description="Bounded, proportional dataset downloader")
    p.add_argument("--out-dir", help="Directory to write <source>/ shards into")
    p.add_argument("--preset", nargs="+", choices=sorted(PRESETS), default=["stage2"],
                   help="One or more budget presets (merged by summing per-source budgets)")
    p.add_argument("--repo", default=DEFAULT_REPO, help="HF dataset repo id")
    p.add_argument("--config-map", default=None,
                   help="JSON mapping of preset source -> dataset config name, for repos "
                        "whose config names differ (e.g. Dolmino)")
    p.add_argument("--tokenizer", default=None,
                   help="Tokenizer path for exact token counts (default: byte heuristic)")
    p.add_argument("--bytes-per-token", type=float, default=4.0,
                   help="Bytes per token when using the heuristic (~4 English, ~3.5 code)")
    p.add_argument("--shard-tokens", type=int, default=500 * 1_000_000,
                   help="Approx tokens per output shard (~2GB at 4 bytes/token)")
    p.add_argument("--scale", type=float, default=1.0,
                   help="Scale all budgets, e.g. 0.01 for a quick smoke test")
    p.add_argument("--max-total-tokens", type=float, default=None,
                   help="Global cap across all sources (in tokens)")
    p.add_argument("--list-presets", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.list_presets:
        for name, budgets in PRESETS.items():
            total = sum(budgets.values())
            print(f"{name}: {total / G:.1f}B tokens")
            for src, b in budgets.items():
                print(f"    {src}: {b / G:.2f}B")
        return

    if not args.out_dir:
        raise SystemExit("error: --out-dir is required unless --list-presets is used")

    config_map = json.loads(args.config_map) if args.config_map else {}
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

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
        total_budget = sum(budgets.values())

    print(f"Presets {args.preset} -> {total_budget / G:.2f}B tokens across {len(budgets)} sources",
          file=sys.stderr)

    total_written = 0
    for source, budget in budgets.items():
        cfg = config_map.get(source, source)
        total_written += download_source(
            args.repo, source, cfg, budget, args.out_dir, tokenizer,
            args.bytes_per_token, args.shard_tokens,
        )

    print(f"Done. Wrote ~{total_written / G:.2f}B tokens into {args.out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
