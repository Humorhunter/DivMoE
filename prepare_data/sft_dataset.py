import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Union, Iterator, Any
import random
from pathlib import Path

import torch
import datasets
from datasets import load_dataset, Dataset, DatasetDict, IterableDatasetDict, IterableDataset
from transformers import PreTrainedTokenizer

from prepare_data.tulu_dataset import load_tulu_dataset

logger = logging.getLogger(__name__)

def build_sft_dataset(
    raw_dataset, 
    tokenizer, 
    block_size=4*1024, 
    logger=None, 
    accelerator=None,
    context_manager=None,
):
    """
    Build a dataset for Supervised Fine-Tuning (SFT) from a raw dataset containing
    conversational data.
    
    This function:
    1. Does not concatenate examples (unlike pretraining)
    2. Sets proper attention mask for loss calculation (only on assistant responses)
    3. Handles multi-turn dialogues by masking losses appropriately
    
    Args:
        raw_dataset: The raw dataset with message sequences
        tokenizer: Tokenizer to use
        block_size: Maximum sequence length
        logger: Logger object
        accelerator: Accelerator for distributed training
        context_manager: Context manager for distributed processing
    
    Returns:
        A processed dataset ready for SFT
    """
    if context_manager is None:
        from contextlib import nullcontext
        context_manager = nullcontext
    
    # Ensure block_size is within tokenizer's limit
    if block_size is None:
        block_size = tokenizer.model_max_length
    else:
        if block_size > tokenizer.model_max_length and logger is not None:
            logger.warning(
                f"The block_size passed ({block_size}) is larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(block_size, tokenizer.model_max_length)
    
    def tokenize_and_mask_function(examples):
        """
        Tokenize conversations and create loss masks for assistant responses only.
        
        This processes each conversation to ensure loss is only calculated on assistant messages.
        """
        # Process each conversation
        all_input_ids = []
        all_attention_mask = []
        all_labels = []
        
        # Iterate through each example
        for messages in examples["messages"]:
            conversation_tokens = []
            conversation_labels = []  # Track which positions should have loss calculated
            
            # Process all messages in a single pass
            for message in messages:
                role = message["role"]
                content = message["content"]
                
                # Tokenize this message
                message_tokens = tokenizer.encode(
                    f"{role}: {content}",
                    add_special_tokens=False  # We'll add special tokens at the sequence level
                )
                
                # Set labels for this message - use actual token IDs for assistant, -100 for others
                if role == "assistant":
                    # For assistant messages, we'll compute loss - use the token IDs
                    message_labels = message_tokens.copy()
                else:
                    # For non-assistant messages, we mask the loss
                    message_labels = [-100] * len(message_tokens)
                
                # Add to the conversation
                conversation_tokens.extend(message_tokens)
                conversation_labels.extend(message_labels)
 
            # Truncate if too long           
            if tokenizer.bos_token_id is not None:
                sentence_delim_tokens = 2
            else:
                sentence_delim_tokens = 1
            if len(conversation_tokens) > block_size - sentence_delim_tokens:  # Account for special tokens
                conversation_tokens = conversation_tokens[:block_size - sentence_delim_tokens]
                conversation_labels = conversation_labels[:block_size - sentence_delim_tokens]

            if sum([each!=-100 for each in conversation_labels]) == 0:
                # all -100 labels lead to NaN loss
                continue

            if tokenizer.bos_token_id is not None:
                # Add special tokens
                input_ids = [tokenizer.bos_token_id] + conversation_tokens + [tokenizer.eos_token_id]
                # Add special token labels (-100 = no loss)
                labels = [-100] + conversation_labels + [-100]  # Don't compute loss on special tokens
            else:
                input_ids = conversation_tokens + [tokenizer.eos_token_id]
                labels = conversation_labels + [-100]  # Don't compute loss on special tokens
            attention_mask = [1] * len(input_ids)  # All tokens get attention

            # Append to batch lists
            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_labels.append(labels)
        
        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": all_labels
        }
    
    with context_manager():
        # Process the dataset directly without concatenation
        return raw_dataset.map(
            tokenize_and_mask_function,
            batched=True,
            remove_columns=raw_dataset.column_names,
        )

def load_sft_dataset(
    tokenizer,
    streaming=True,
    split="train",
    block_size=4*1024,
    logger=None,
    accelerator=None,
    seed=None,
    use_tulu=True,
):
    """
    Load and prepare a dataset for Supervised Fine-Tuning
    
    Args:
        tokenizer: The tokenizer to use
        streaming: Whether to stream the dataset
        split: Dataset split to load (typically "train")
        block_size: Maximum sequence length
        logger: Logger object 
        accelerator: Accelerator for distributed training
        seed: Random seed
        use_tulu: Whether to use the Tulu dataset (default) or another dataset
    
    Returns:
        Processed SFT dataset ready for training
    """
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    
    if logger is not None:
        logger.info(f"Loading SFT dataset (streaming={streaming})")
    
    # Load raw dataset
    if use_tulu:
        raw_dataset = load_tulu_dataset(
            streaming=streaming,
            split=split,
            seed=seed,
        )
    else:
        # Could add support for other datasets here
        raise ValueError("Currently only Tulu dataset is supported")
    
    # Create a context manager for distributed processing
    if accelerator is not None:
        context_manager = accelerator.main_process_first
    else:
        from contextlib import nullcontext
        context_manager = nullcontext
    
    # Build the SFT dataset
    processed_dataset = build_sft_dataset(
        raw_dataset[split], 
        tokenizer, 
        block_size=block_size,
        logger=logger,
        accelerator=accelerator,
        context_manager=context_manager,
    )
    
    # Wrap in appropriate dictionary
    if streaming:
        return {split: processed_dataset}
    else:
        return DatasetDict({split: processed_dataset})

def test_sft_dataset_loading(tokenizer_name_or_path):
    """Test function to verify SFT dataset loading works."""
    from transformers import AutoTokenizer
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    
    dataset = load_sft_dataset(
        tokenizer=tokenizer,
        streaming=False,  # Use non-streaming for testing
        logger=logger,
    )
    
    logger.info(f"Dataset loaded successfully: {dataset}")
    
    # Print a sample
    example = next(iter(dataset["train"]))
    logger.info(f"Example input_ids shape: {len(example['input_ids'])}")
    logger.info(f"Example attention_mask shape: {len(example['attention_mask'])}")
    logger.info(f"Example labels shape: {len(example['labels'])}")
    
    # Show where loss is calculated
    non_masked_positions = [i for i, label in enumerate(example['labels']) if label != -100]
    logger.info(f"Loss calculated on {len(non_masked_positions)} / {len(example['labels'])} tokens")
    
    # Decode some parts to verify
    input_text = tokenizer.decode(example['input_ids'])
    logger.info(f"Input text snippet: {input_text[:200]}...")
    
    # Decode assistant parts where loss is calculated
    assistant_text = tokenizer.decode([example['input_ids'][i] for i in non_masked_positions])
    logger.info(f"Assistant text (loss calculated): {assistant_text[:200]}...")


def cache_sft_dataset(tokenizer_name_or_path, cache_to_disk_path):
    from transformers import AutoTokenizer
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    
    dataset = load_sft_dataset(
        tokenizer=tokenizer,
        streaming=False,  # Non-streaming to enable shuffle and use with DataLoader
        split="train",
        block_size=4096,
        logger=logger,
        accelerator=None,
        seed=42,
        use_tulu=True,
    )

    dataset.save_to_disk(cache_to_disk_path)


if __name__ == "__main__":
    pass
    # import sys
    
    # if len(sys.argv) > 1:
    #     tokenizer_name = sys.argv[1]
    # else:
    #     tokenizer_name = "mistralai/Mistral-7B-v0.1"
    
    # logging.basicConfig(level=logging.INFO)
    # test_sft_dataset_loading(tokenizer_name) 

    # Example usage:
    # cache_sft_dataset("/path/to/tokenizer", "/path/to/output")
