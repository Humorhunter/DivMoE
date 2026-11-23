import os
import io
import json
import glob
from pathlib import Path
import datasets
from datasets import load_dataset, Dataset, IterableDatasetDict, IterableDataset, Features, Value
import zstandard as zstd
import gzip
from typing import Dict, List, Optional, Tuple, Union, Iterator, Any
import logging
import random
from collections import defaultdict
import numpy as np
from itertools import cycle, islice
import torch
from torch.utils.data import get_worker_info
from datasets.utils.py_utils import size_str
from datasets.iterable_dataset import ExamplesIterable

logger = logging.getLogger(__name__)

_CITATION = """
@misc{muennighoff2024olmoeopenmixtureofexpertslanguage,
      title={OLMoE: Open Mixture-of-Experts Language Models}, 
      author={Niklas Muennighoff and Luca Soldaini and Dirk Groeneveld and Kyle Lo and Jacob Morrison and Sewon Min and Weijia Shi and Pete Walsh and Oyvind Tafjord and Nathan Lambert and Yuling Gu and Shane Arora and Akshita Bhagia and Dustin Schwenk and David Wadden and Alexander Wettig and Binyuan Hui and Tim Dettmers and Douwe Kiela and Ali Farhadi and Noah A. Smith and Pang Wei Koh and Amanpreet Singh and Hannaneh Hajishirzi},
      year={2024},
      eprint={2409.02060},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.02060}, 
}
"""

_DESCRIPTION = """
Fixed OLMoE Mix dataset with stable training behavior and good mixing properties.
"""

class FixedOLMoEExamplesIterable(ExamplesIterable):
    """Fixed ExamplesIterable that addresses training instability issues"""
    
    def __init__(
        self,
        base_dir: str,
        buffer_size: int,
        features: Features,
        source_props: Dict[str, float],
        seed: Optional[int] = None,
        global_step: int = 0
    ):
        super().__init__(iter([]), features)
        self.base_dir = base_dir
        self.buffer_size = buffer_size
        self.source_props = source_props
        self.seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        self.global_step = global_step
        self._example_idx = 0
        
        # Pre-discover all files for each source
        self.source_files = {}
        for source in self.source_props.keys():
            source_dir = os.path.join(self.base_dir, source)
            files = glob.glob(os.path.join(source_dir, "**/*.json.*"), recursive=True)
            if files:
                self.source_files[source] = files
                logger.info(f"Found {len(files)} files for source '{source}'")
            else:
                logger.warning(f"No files found for source '{source}' in {source_dir}")
    
    def _get_iterator_for_worker(
        self,
        worker_id: Optional[int],
        num_workers: Optional[int]
    ) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """Worker-aware iterator with improved stability"""
        
        # Set worker-specific seed but keep it simple and predictable
        if worker_id is not None and num_workers is not None:
            worker_seed = self.seed + worker_id * 1000 + self.global_step
            self._example_idx = worker_id * 1000000  # Offset for each worker
        else:
            worker_seed = self.seed + self.global_step
        
        # Set random state
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        
        # Create worker-specific file lists (but don't partition too aggressively)
        worker_source_files = {}
        for source, files in self.source_files.items():
            if num_workers and worker_id is not None and num_workers > 1:
                # Simple round-robin distribution instead of slicing
                worker_files = [f for i, f in enumerate(files) if i % num_workers == worker_id]
            else:
                worker_files = files
            
            if worker_files:
                worker_source_files[source] = worker_files
        
        if not worker_source_files:
            logger.warning("No files available for this worker")
            return
        
        # Initialize simple deterministic source selection (no complex mixer)
        sources = list(worker_source_files.keys())
        weights = [self.source_props[s] for s in sources]
        
        # Pre-compute a stable sequence of source selections
        source_rng = np.random.RandomState(worker_seed)
        sequence_length = 50000  # Smaller, more predictable sequence
        source_sequence = source_rng.choice(sources, size=sequence_length, p=weights)
        source_idx = 0
        
        # Initialize file iterators for each source
        source_iterators = {}
        for source, files in worker_source_files.items():
            # Shuffle files deterministically 
            file_rng = np.random.RandomState(worker_seed + hash(source) % 10000)
            shuffled_files = files.copy()
            file_rng.shuffle(shuffled_files)
            source_iterators[source] = self._create_file_iterator(shuffled_files)
        
        # Initialize buffers with larger size for better mixing
        buffers = {source: [] for source in worker_source_files.keys()}
        
        while True:
            # Get next source from pre-computed sequence
            source = source_sequence[source_idx % sequence_length]
            source_idx += 1
            
            if source not in source_iterators:
                continue
                
            # Fill buffer if needed
            if len(buffers[source]) < 1:
                try:
                    # Fill buffer with multiple items at once
                    for _ in range(min(self.buffer_size // 10, 1000)):  # Fill chunk by chunk
                        try:
                            item = next(source_iterators[source])
                            item["source"] = source
                            buffers[source].append(item)
                        except StopIteration:
                            # Recreate iterator when exhausted
                            files = worker_source_files[source]
                            file_rng = np.random.RandomState(worker_seed + hash(source) + source_idx)
                            file_rng.shuffle(files)
                            source_iterators[source] = self._create_file_iterator(files)
                            break
                    
                    # Shuffle buffer for better mixing
                    if buffers[source]:
                        buffer_rng = np.random.RandomState(worker_seed + self._example_idx)
                        buffer_rng.shuffle(buffers[source])
                    
                except Exception as e:
                    logger.warning(f"Error filling buffer for source {source}: {e}")
                    continue
            
            if buffers[source]:
                example = buffers[source].pop()
                key = f"{source}-worker{worker_id or 0}-{self._example_idx}"
                self._example_idx += 1
                yield key, example

    def _create_file_iterator(self, files: List[str]) -> Iterator[Dict[str, Any]]:
        """Create an iterator over files"""
        for file in cycle(files):  # Use cycle for infinite iteration
            if file.endswith('.json.zst'):
                yield from self._read_zst_file(file)
            elif file.endswith('.json.gz'):
                yield from self._read_gz_file(file)
    
    def _read_zst_file(self, filepath: str) -> Iterator[Dict[str, Any]]:
        """Read a zstandard compressed JSON file"""
        try:
            with open(filepath, "rb") as f:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(f) as reader:
                    for line in io.TextIOWrapper(reader, encoding='utf-8'):
                        if line.strip():
                            try:
                                example = json.loads(line)
                                # Normalize field names
                                if "text" not in example and "content" in example:
                                    example["text"] = example["content"]
                                if "id" not in example:
                                    example["id"] = ""
                                if "added" not in example:
                                    example["added"] = ""
                                if "created" not in example:
                                    example["created"] = ""
                                yield example
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            logger.warning(f"Error reading file {filepath}: {str(e)}")
            return
    
    def _read_gz_file(self, filepath: str) -> Iterator[Dict[str, Any]]:
        """Read a gzip compressed JSON file"""
        try:
            with gzip.open(filepath, "rt", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            example = json.loads(line)
                            # Normalize field names
                            if "text" not in example and "content" in example:
                                example["text"] = example["content"]
                            if "id" not in example:
                                example["id"] = ""
                            if "added" not in example:
                                example["added"] = ""
                            if "created" not in example:
                                example["created"] = ""
                            yield example
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.warning(f"Error reading file {filepath}: {str(e)}")
            return

    def __iter__(self):
        """Return worker-aware iterator"""
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else None
        num_workers = worker_info.num_workers if worker_info else None
        
        return self._get_iterator_for_worker(worker_id, num_workers)

class FixedProportionalMixIterableDataset(IterableDataset):
    """
    Fixed version that addresses training instability issues:
    - Larger buffer size for better shuffling
    - Simpler, more predictable source selection
    - No complex threading or race conditions
    - Better worker coordination
    """
    def __init__(
        self, 
        base_dir: str, 
        buffer_size: int = 100_000,  # Back to original larger size
        streaming: bool = True,
        seed: Optional[int] = None,
        global_step: int = 0
    ):
        # Source proportions based on OLMoE paper
        source_props = {
            "dclm": 0.40,
            "starcoder": 0.15,
            "pes2o": 0.15,
            "algebraic-stack": 0.10,
            "open-web-math": 0.10,
            "wiki": 0.10
        }
        
        features = Features({
            "text": Value("string"),
            "id": Value("string"),
            "added": Value("string"),
            "created": Value("string"),
            "source": Value("string")
        })

        info = datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage="https://huggingface.co/datasets/allenai/OLMoE-mix-0924",
            citation=_CITATION,
        )
        
        # Initialize with fixed iterable
        super().__init__(
            FixedOLMoEExamplesIterable(
                base_dir=base_dir,
                buffer_size=buffer_size,
                features=features,
                source_props=source_props,
                seed=seed,
                global_step=global_step,
            ),
            info=info,
        )
        
        self.base_dir = base_dir
        self.buffer_size = buffer_size
        self.streaming = streaming
        self.seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        self.global_step = global_step
        self.source_props = source_props
        self._features = features

    @property
    def features(self):
        return self._features
    
    def with_global_step(self, global_step: int):
        """Create a new dataset instance with updated global step for deterministic resuming"""
        return FixedProportionalMixIterableDataset(
            base_dir=self.base_dir,
            buffer_size=self.buffer_size,
            streaming=self.streaming,
            seed=self.seed,
            global_step=global_step
        )

def load_fixed_olmoe_mix_dataset(
    data_dir: str,
    buffer_size: int = 100_000,  # Back to original size
    streaming: bool = True,
    seed: Optional[int] = None,
    global_step: int = 0
) -> IterableDatasetDict:
    """
    Load the fixed OLMoE mix dataset with stable training behavior.
    
    Args:
        data_dir: Directory containing the OLMoE mix data files
        buffer_size: Size of the buffer for each category for shuffling  
        streaming: Whether to use streaming mode
        seed: Random seed for reproducibility
        global_step: Current global step for deterministic resuming
    
    Returns:
        A HuggingFace IterableDatasetDict with stable training behavior
    """
    train_dataset = FixedProportionalMixIterableDataset(
        base_dir=data_dir,
        buffer_size=buffer_size,
        streaming=streaming,
        seed=seed,
        global_step=global_step
    )
    
    return IterableDatasetDict({
        "train": train_dataset
    })

def test_fixed_dataset_loading(data_dir: str):
    """Test the fixed dataset loading"""
    try:
        print(f"Testing fixed dataset loading from {data_dir}")
        
        dataset = load_fixed_olmoe_mix_dataset(data_dir, buffer_size=100000)
        
        print("\nGetting sample examples...")
        source_counts = defaultdict(int)
        
        for i, example in enumerate(dataset["train"]):
            if i == 0:
                print(f"First example fields: {list(example.keys())}")
                print(f"Text preview: {example['text'][:100]}...")
            
            source_counts[example['source']] += 1
            
            if i >= 999:  # Test with 1000 examples
                break
        
        print("\nSource distribution:")
        total = sum(source_counts.values())
        for source, count in sorted(source_counts.items()):
            percentage = count / total * 100
            print(f"{source}: {count} examples ({percentage:.1f}%)")
            
        print("\nFixed dataset test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error testing fixed dataset: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python olmoe_dataset_fixed.py <data_directory>")
        sys.exit(1)
        
    data_dir = sys.argv[1]
    success = test_fixed_dataset_loading(data_dir)
    sys.exit(0 if success else 1) 