import json
import logging
from itertools import chain
import torch
from torch.utils.data import IterableDataset
import re
import transformers

from datasets import load_dataset, interleave_datasets

logger = logging.getLogger(__name__)

DEFAULT_CHAT_TEMPLATE = "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\n'  + message['content'] + eos_token }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}{% endfor %}"


def _format_and_tokenize(example, tokenizer, include_thinking=True):
    """
    Formats a single SFT example using the tokenizer's chat template,
    tokenizes it, and creates labels with prompt masking.
    """
    # Set a default chat template if the tokenizer doesn't have one
    if tokenizer.chat_template is None:
        logger.warning("Tokenizer does not have a chat_template, using a default template.")
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    # 1. Parse the input into a list of messages (the prompt)
    try:
        if isinstance(example['input'], str) and example['input']:
            prompt_messages = json.loads(example['input'])
        else:
            prompt_messages = example.get('input') or []
    except (json.JSONDecodeError, TypeError):
        prompt_messages = [{"role": "user", "content": str(example.get('input'))}]

    # 2. Get the assistant's response
    response_str = example.get('output', '') or ""
    if not response_str: # Skip examples with no response
        return None, None

    # Optionally remove content within <think>...</think> tags
    if not include_thinking:
        response_str = re.sub(r'<think>.*?</think>\s*', '', response_str, flags=re.DOTALL).strip()

    # 3. Apply the chat template to the prompt part.
    # We add the generation prompt to get the tokens that lead up to the assistant's response.
    prompt_tokens = tokenizer.apply_chat_template(
        prompt_messages, 
        add_generation_prompt=True, 
        tokenize=True, 
        add_special_tokens=False
    )

    # 4. Apply the chat template to the full conversation.
    full_messages = prompt_messages + [{"role": "assistant", "content": response_str}]
    full_tokens = tokenizer.apply_chat_template(
        full_messages, 
        tokenize=True, 
        add_special_tokens=False
    )
    
    # The chat template should handle the EOS token. No manual append needed.

    # 5. Create labels by masking the prompt part.
    labels = [-100] * len(prompt_tokens) + full_tokens[len(prompt_tokens):]

    # Final input_ids are the full token sequence.
    input_ids = full_tokens
    
    if len(input_ids) != len(labels):
        logger.warning("Mismatch between input_ids and labels length after applying chat template. Skipping example.")
        return None, None
        
    return input_ids, labels


class PackedSFTDataset(IterableDataset):
    """
    An iterable dataset that packs sequences for SFT.
    This class wraps the sequence packing generator to make it compatible with PyTorch's DataLoader.
    """
    def __init__(self, dataset_name, config_name, tokenizer, streaming, seed, split, block_size, include_thinking=True, num_validation_samples=None):
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.tokenizer = tokenizer
        self.streaming = streaming
        self.seed = seed
        self.split = split
        self.block_size = block_size
        self.include_thinking = include_thinking
        self.num_validation_samples = num_validation_samples

    def __iter__(self):
        # Load the raw dataset, handling multiple splits if necessary
        if isinstance(self.split, list):
            datasets_to_interleave = [
                load_dataset(self.dataset_name, name=self.config_name, split=s, streaming=self.streaming) for s in self.split
            ]
            raw_dataset = interleave_datasets(datasets_to_interleave, seed=self.seed)
        else:
            raw_dataset = load_dataset(self.dataset_name, name=self.config_name, split=self.split, streaming=self.streaming)

        # For creating a validation set from the training data
        if self.num_validation_samples:
            raw_dataset = raw_dataset.shuffle(seed=self.seed, buffer_size=10_000).take(self.num_validation_samples)
        
        buffer_input_ids = []
        buffer_labels = []

        for example in raw_dataset:
            input_ids, labels = _format_and_tokenize(example, self.tokenizer, self.include_thinking)
            if input_ids is None:
                continue
            
            buffer_input_ids.extend(input_ids)
            buffer_labels.extend(labels)

            while len(buffer_input_ids) >= self.block_size:
                packed_input_ids = buffer_input_ids[:self.block_size]
                packed_labels = buffer_labels[:self.block_size]

                yield {
                    "input_ids": torch.tensor(packed_input_ids, dtype=torch.long),
                    "labels": torch.tensor(packed_labels, dtype=torch.long),
                    "attention_mask": torch.ones(self.block_size, dtype=torch.long)
                }
                
                # Carry over the remainder to the next iteration
                buffer_input_ids = buffer_input_ids[self.block_size:]
                buffer_labels = buffer_labels[self.block_size:]


def build_packed_sft_dataset(
    dataset_name, config_name, tokenizer, streaming=True, seed=None, 
    split='train', block_size=32768, include_thinking=False, num_validation_samples=None
):
    """
    Builds an IterableDataset for SFT with sequence packing.
    """
    logger.info(f"Building packed SFT IterableDataset for split '{split}' with block size {block_size}")
    return PackedSFTDataset(
        dataset_name, config_name, tokenizer, streaming, seed, split, block_size, include_thinking, num_validation_samples
    )


if __name__ == '__main__':
    # This is a test block to demonstrate and verify the dataset building.
    # You would typically use a real model tokenizer, e.g., from 'meta-llama/Llama-2-7b-hf'
    # For demonstration, we use a dummy tokenizer.
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # 1. Setup a tokenizer
    # In a real scenario, you would load your model's tokenizer.
    # For this test, we'll use a pretrained one that has a chat template.
    tokenizer = transformers.AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
    
    # 2. Define dataset parameters
    # Using a small, public dataset for demonstration
    dataset_name = "HuggingFaceH4/ultrachat_200k"
    config_name = "default" # 'default' is often used for datasets without multiple configs
    split = "train_sft"
    block_size = 1024
    seed = 42

    # 3. Build the dataset with thinking enabled (default)
    print("\n--- Testing dataset with thinking patterns included ---")
    sft_dataset_with_thinking = build_packed_sft_dataset(
        dataset_name=dataset_name,
        config_name=config_name,
        tokenizer=tokenizer,
        split=split,
        block_size=block_size,
        seed=seed,
        include_thinking=True, # Explicitly True
    )

    # 4. Iterate and print a sample
    print("Fetching one sample...")
    for i, sample in enumerate(sft_dataset_with_thinking):
        print(f"Sample {i+1} (with thinking):")
        print("input_ids shape:", sample['input_ids'].shape)
        print("labels shape:", sample['labels'].shape)
        # To see the actual text, you would decode the tensors.
        # Note: -100 in labels is for masked tokens (the prompt).
        # print("Decoded input_ids:", tokenizer.decode(sample['input_ids']))
        if i == 0: # Print only the first sample
            break
    
    # 5. Build the dataset with thinking disabled
    print("\n--- Testing dataset with thinking patterns excluded ---")
    # To test this properly, you need a dataset with <think> tags.
    # Since ultrachat doesn't have them, this demonstrates the flag is passed,
    # but won't change the output for this specific dataset.
    sft_dataset_no_thinking = build_packed_sft_dataset(
        dataset_name=dataset_name,
        config_name=config_name,
        tokenizer=tokenizer,
        split=split,
        block_size=block_size,
        seed=seed,
        include_thinking=False,
    )

    print("Fetching one sample...")
    for i, sample in enumerate(sft_dataset_no_thinking):
        print(f"Sample {i+1} (no thinking):")
        print("input_ids shape:", sample['input_ids'].shape)
        print("labels shape:", sample['labels'].shape)
        if i == 0:
            break