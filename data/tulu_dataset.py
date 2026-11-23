import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Union, Iterator, Any
import random
from pathlib import Path

import datasets
from datasets import load_dataset, Dataset, DatasetDict, IterableDatasetDict, IterableDataset, Features, Value

logger = logging.getLogger(__name__)

_CITATION = """
@misc{tulu3sftmixture,
      title={Tulu-3-sft-mixture}, 
      author={AI2 Team},
      year={2023},
      url={https://huggingface.co/datasets/allenai/tulu-3-sft-mixture},
}
"""

_DESCRIPTION = """
Tulu-3-sft-mixture is an instruction-following dataset designed for Supervised Fine-tuning (SFT)
of language models. It combines various high-quality instruction datasets to create a comprehensive
mixture for training models to follow human instructions effectively.
"""

class TuluDatasetBuilder(datasets.GeneratorBasedBuilder):
    """Tulu-3-sft-mixture dataset for instruction-tuning language models."""

    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="default",
            version=VERSION,
            description="Tulu-3-sft-mixture dataset for instruction-tuning language models",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "messages": datasets.Sequence(
                        {
                            "role": datasets.Value("string"),
                            "content": datasets.Value("string"),
                        }
                    ),
                    "id": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="https://huggingface.co/datasets/allenai/tulu-3-sft-mixture",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"split": "train"},
            ),
        ]

    def _generate_examples(self, split):
        """Yields examples."""
        try:
            # Load the Tulu-3-sft-mixture dataset from Hugging Face
            tulu_dataset = load_dataset("allenai/tulu-3-sft-mixture", split=split)
            
            for example_id, example in enumerate(tulu_dataset):
                # Format the example to match expected structure
                # (Tulu dataset has a "messages" field with role and content)
                yield example_id, {
                    "messages": example["messages"],
                    "id": example.get("id", str(example_id)),
                }
        except Exception as e:
            logger.error(f"Error loading Tulu dataset: {e}")
            raise

def format_for_training(example, tokenizer, message_template=None):
    """Format messages into a format suitable for model training.
    
    Args:
        example: The dataset example with a "messages" field
        tokenizer: The tokenizer to use
        message_template: Optional template for formatting messages
        
    Returns:
        A dictionary with "text" field containing the formatted conversation
    """
    messages = example["messages"]
    
    # Default template (similar to ChatML format)
    if message_template is None:
        formatted_text = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                formatted_text += f"<|user|>\n{content}\n"
            elif role == "assistant":
                formatted_text += f"<|assistant|>\n{content}\n"
            elif role == "system":
                formatted_text += f"<|system|>\n{content}\n"
        
        # Ensure we end with an assistant message for training
        if messages and messages[-1]["role"] != "assistant":
            formatted_text += "<|assistant|>\n"
    else:
        # Use custom template if provided
        formatted_text = message_template(messages)
    
    return {"text": formatted_text}

def load_tulu_dataset(
    streaming: bool = True,
    split: str = "train",
    format_function=None,
    tokenizer=None,
    message_template=None,
    seed: Optional[int] = None,
) -> Union[Dataset, IterableDataset]:
    """Load the Tulu-3-sft-mixture dataset.
    
    Args:
        streaming: Whether to stream the dataset
        split: Dataset split to load ("train")
        format_function: Function to format the example for training
        tokenizer: Tokenizer for formatting (if needed)
        message_template: Template for formatting messages
        seed: Random seed
        
    Returns:
        Dataset or IterableDataset with the Tulu data
    """
    logger.info(f"Loading Tulu-3-sft-mixture dataset (streaming={streaming})")
    
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    
    # Load dataset from Hugging Face
    tulu_dataset = load_dataset(
        # "allenai/tulu-3-sft-mixture",
        "/path/to/tulu-3-sft-mixture",
        split=split,
        streaming=streaming,
    )
    
    # Apply formatting if function is provided
    if format_function and tokenizer:
        logger.info("Applying custom formatting to dataset")
        tulu_dataset = tulu_dataset.map(
            lambda example: format_function(example, tokenizer, message_template)
        )
    elif tokenizer:  # Use default formatting
        logger.info("Applying default formatting to dataset")
        tulu_dataset = tulu_dataset.map(
            lambda example: format_for_training(example, tokenizer, message_template)
        )
    
    # Wrap in DatasetDict or IterableDatasetDict
    if streaming:
        return IterableDatasetDict({split: tulu_dataset})
    else:
        return DatasetDict({split: tulu_dataset})

def test_dataset_loading():
    """Test function to verify dataset loading works."""
    dataset = load_tulu_dataset(streaming=True)
    print(f"Dataset loaded successfully: {dataset}")
    
    # Print a sample
    for i, example in enumerate(dataset["train"]):
        print(f"Example {i}:")
        print(example)
        if i >= 2:  # Just show a few examples
            break

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_dataset_loading()