#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning models for instruction following tasks with Supervised Fine-Tuning (SFT)
without using HuggingFace Trainer.

This script is adapted from the causal language modeling script for SFT tasks.
The main differences are:
1. No concatenation of examples - each conversation is a separate training example
2. Loss masking - only calculating loss on assistant responses
3. Support for multi-turn conversations
4. Knowledge distillation - option to distill from a teacher model (use --distillation flag)

Example usage with distillation:
python run_sft_no_trainer.py 
  --model_name_or_path EleutherAI/pythia-1.4b 
  --teacher_model_name_or_path mistralai/Mistral-7B-v0.1 
  --distillation 
  --distillation_alpha 0.5 
  --distillation_temperature 2.0
  --output_dir ./sft_distilled_model
"""

import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path
from contextlib import nullcontext
from datetime import timedelta

import datasets
import torch
import torch.nn.functional as F
from accelerate import Accelerator, DeepSpeedPlugin, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset, DatasetDict, load_from_disk
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Import the SFT dataset loader
from prepare_data.sft_dataset import load_sft_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    DataCollatorForTokenClassification,
    get_scheduler,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

# Import local DivMoE model for training if available
try:
    from modeling_divmoe import DivMoEConfig, DivMoEForCausalLM
except ImportError:
    pass  # DivMoE model is optional

logger = get_logger(__name__)

require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model with Supervised Fine-Tuning (SFT)")
    parser.add_argument(
        "--sft_cache_path", type=str, default=None
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, 
        help="A csv, txt or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--min_lr", type=float, default=None,
        help="Minimum learning rate for cosine scheduler."
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        # choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--use_tulu_dataset", action="store_true", default=False,
        help="Use the Tulu dataset for SFT training."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help=(
            "Whether to trust the execution of code from datasets/models defined on the Hub."
            " This option should only be set to `True` for repositories you trust and in which you have read the"
            " code, as it will execute code present on the Hub on your local machine."
        ),
    )
    # Distillation arguments
    parser.add_argument(
        "--distillation", action="store_true", 
        help="Whether to use knowledge distillation during training."
    )
    parser.add_argument(
        "--teacher_model_name_or_path", type=str, default=None,
        help="Path to teacher model or model identifier from huggingface.co/models for distillation."
    )
    parser.add_argument(
        "--distillation_alpha", type=float, default=0.5,
        help="Weight for distillation loss (0 to 1). Higher values put more weight on matching teacher outputs."
    )
    parser.add_argument(
        "--distillation_temperature", type=float, default=2.0,
        help="Temperature for softening probability distributions in distillation."
    )
    parser.add_argument(
        "--use_divmoe",
        action="store_true",
        help="Whether to use the DivMoE model architecture defined in modeling_divmoe.py.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=0,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument("--evaluate_every", type=int, default=100)
    parser.add_argument("--skip_train", action='store_true')
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"swanlab"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    args = parser.parse_args()

    # Sanity checks
    if args.sft_cache_path is None and args.dataset_name is None and args.validation_file is None and not args.use_tulu_dataset:
        raise ValueError("Need either --sft_cache_path, --dataset_name, --validation_file, or --use_tulu_dataset.")
    else:
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`validation_file` should be a csv, json or txt file.")

    if args.push_to_hub:
        if args.output_dir is None:
            raise ValueError("Need an `output_dir` to create a repo when `--push_to_hub` is passed.")

    # Validate DivMoE related arguments
    if args.use_divmoe:
        if 'DivMoEForCausalLM' not in globals():
            raise ImportError("DivMoE model not found. Make sure modeling_divmoe.py is in the same directory.")
        if args.model_name_or_path is None:
            raise ValueError("--model_name_or_path must be provided as the base model when using DivMoE.")
        
        AutoConfig.register("divmoe", DivMoEConfig, exist_ok=True)
    
    # Validate distillation related arguments
    if args.distillation:
        if args.teacher_model_name_or_path is None:
            raise ValueError("--teacher_model_name_or_path must be provided when using distillation.")
        if not (0 <= args.distillation_alpha <= 1):
            raise ValueError("--distillation_alpha must be between 0 and 1.")
        if args.distillation_temperature <= 0:
            raise ValueError("--distillation_temperature must be positive.")

    return args

GB = 1024**3

def get_memory_stats():
    """Get GPU memory statistics"""
    alloc = torch.cuda.memory_allocated() / GB
    max_alloc = torch.cuda.max_memory_allocated() / GB
    reserved = torch.cuda.memory_reserved() / GB
    max_reserved = torch.cuda.max_memory_reserved() / GB
    return alloc, max_alloc, reserved, max_reserved 

def compute_distillation_loss(student_logits, teacher_logits, temperature=2.0):
    """
    Compute the knowledge distillation loss between student and teacher logits.
    
    Args:
        student_logits: Logits from the student model (shape: batch_size x seq_len x vocab_size)
        teacher_logits: Logits from the teacher model (shape: batch_size x seq_len x vocab_size)
        temperature: Temperature for softening probability distributions
        
    Returns:
        KL divergence loss between softened distributions
    """
    # Apply temperature scaling to soften the distributions
    soft_student_logits = student_logits / temperature
    soft_teacher_logits = teacher_logits.detach() / temperature  # Detach to prevent gradient flow to teacher
    
    # Compute KL divergence loss
    # We use F.kl_div with log_softmax applied to student and softmax to teacher
    # reduction='batchmean' to get scalar value
    # Note: input to kl_div should be log probabilities, target should be probabilities
    loss_kl = F.kl_div(
        input=F.log_softmax(soft_student_logits, dim=-1),
        target=F.softmax(soft_teacher_logits, dim=-1),
        reduction='batchmean',
        log_target=False,
    )
    
    # Multiply by temperature^2 to scale gradients back up 
    # (since we divided by temperature earlier)
    return loss_kl * (temperature ** 2)

def main():
    args = parse_args()

    #################
    # Initialize the accelerator
    accelerator_log_kwargs = {}
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    # Configure DeepSpeed plugins
    deepspeed_plugin = None
    deepspeed_plugins_dict = None
    if args.distillation:
        # Student plugin (ZeRO-2 for training)
        student_plugin = DeepSpeedPlugin(
            gradient_clipping=1.0,
            zero_stage=2,
            zero3_save_16bit_model=True,  # Keep ZeRO-2 settings for student
        )
        # Teacher plugin (ZeRO-3 for inference - minimal config)
        # Create a minimal config dict for ZeRO-3 inference
        # We assume bf16 is used based on the accelerator config below
        teacher_ds_config = {
            "bf16": {"enabled": True},
            "zero_optimization": {
                "stage": 3,
                "overlap_comm": True,
                # "reduce_bucket_size": "auto",
                # "stage3_prefetch_bucket_size": "auto",
                # "stage3_param_persistence_threshold": "auto",
                # "stage3_max_live_parameters": "auto",
                # "stage3_max_reuse_distance": "auto",
            },
            "gradient_clipping": 1.0, 
            "train_micro_batch_size_per_gpu": 1, 
        }
        teacher_plugin = DeepSpeedPlugin(hf_ds_config=teacher_ds_config)
        
        deepspeed_plugins_dict = {"student": student_plugin, "teacher": teacher_plugin}
        # # The accelerator will use the first plugin in the dict ("student") by default
        # deepspeed_plugin = deepspeed_plugins_dict # Pass the dict here

        init_process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=10800))
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            # Pass either the single plugin or the dict of plugins
            deepspeed_plugins=deepspeed_plugins_dict, 
            mixed_precision="bf16",
            kwargs_handlers=[init_process_group_kwargs],
            **accelerator_log_kwargs
        )
    else:
        # Default single plugin setup if not distilling
        deepspeed_plugin = DeepSpeedPlugin(
            gradient_clipping=1.0,
            zero_stage=2,
            zero3_save_16bit_model=True,
        )

        init_process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=10800))
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            # Pass either the single plugin or the dict of plugins
            deepspeed_plugin=deepspeed_plugin, 
            mixed_precision="bf16",
            kwargs_handlers=[init_process_group_kwargs],
            **accelerator_log_kwargs
        )
    
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(f"{accelerator.state}\n rank: {accelerator.process_index}/{accelerator.num_processes}: "
                f"is_main_process: {accelerator.is_main_process}, is_local_main_process: {accelerator.is_local_main_process}", 
                main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Set seed for reproducibility
    if args.seed is not None:
        set_seed(args.seed)
    
    # Create output directory
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    #################
    # Prepare model & tokenizer
    if args.use_divmoe:
        # Check if DivMoE is available
        if 'DivMoEForCausalLM' not in globals():
            raise ImportError(
                "DivMoE model not found. Make sure modeling_divmoe.py is in the same directory."
            )
        
        # Initialize DivMoE model from original model path
        logger.info(f"Initializing DivMoE model from {args.model_name_or_path}")
        model = DivMoEForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16)
        logger.info(f"DivMoE model initialized from {args.model_name_or_path}")
    else:
        # Load the model
        if args.config_name:
            config = AutoConfig.from_pretrained(args.config_name, trust_remote_code=args.trust_remote_code)
        elif args.model_name_or_path:
            config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
        else:
            config = CONFIG_MAPPING[args.model_type]()
            logger.warning("You are instantiating a new config instance from scratch.")

        if args.model_name_or_path:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                low_cpu_mem_usage=args.low_cpu_mem_usage,
                trust_remote_code=args.trust_remote_code,
                torch_dtype=torch.bfloat16,
            )
        else:
            logger.info("Training new model from scratch")
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=args.trust_remote_code)

    # Load tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    else:
        raise ValueError("You must provide a tokenizer with --tokenizer_name or --model_name_or_path")

    # Check model parameters and memory usage
    alloc, max_alloc, reserved, max_reserved = get_memory_stats()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model parameters: {trainable_params/1024**3:.2f} B, device: {model.device}, dtype: {model.dtype}"
        f", Memory stats after initializing model: Alloc: {alloc:.2f} G / {max_alloc:.2f} G, Resrv: {reserved:.2f} G / {max_reserved:.2f} G", 
        main_process_only=False)
    
    # Load teacher model for distillation if enabled
    teacher_model = None
    if args.distillation:
        logger.info(f"Loading teacher model from {args.teacher_model_name_or_path} for distillation")
        teacher_config = AutoConfig.from_pretrained(
            args.teacher_model_name_or_path, trust_remote_code=args.trust_remote_code
        )
        teacher_model = AutoModelForCausalLM.from_pretrained(
            args.teacher_model_name_or_path,
            config=teacher_config,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=torch.bfloat16,
        )
        # Set teacher to evaluation mode
        teacher_model.eval()
        
        # Count teacher parameters
        teacher_params = sum(p.numel() for p in teacher_model.parameters())
        logger.info(
            f"Teacher model loaded with {teacher_params/1024**3:.2f} B parameters, {trainable_params/1024**3:.2f} B student parameters"
        )
        
        # Memory stats after loading teacher
        alloc, max_alloc, reserved, max_reserved = get_memory_stats()
        logger.info(
            f"Memory stats after loading teacher model: Alloc: {alloc:.2f} G / {max_alloc:.2f} G, Resrv: {reserved:.2f} G / {max_reserved:.2f} G", 
            main_process_only=False)
    
    # Resize embeddings if necessary
    embedding_size = model.get_input_embeddings().weight.shape[0]
    max_embedding_size = max(embedding_size, len(tokenizer))
    if teacher_model:
        max_embedding_size = max(max_embedding_size, teacher_model.get_input_embeddings().weight.shape[0])
    if max_embedding_size > embedding_size:
        model.resize_token_embeddings(max_embedding_size)
        logger.info(f"Resized token embeddings from {embedding_size} to {max_embedding_size}")
    
    #################
    # Load SFT dataset
    logger.info(f"Loading SFT dataset (using Tulu: {args.use_tulu_dataset})")
    
    # Load training dataset
    if args.sft_cache_path:
        train_dataset = load_from_disk(args.sft_cache_path)
    else:
        train_dataset = load_sft_dataset(
            tokenizer=tokenizer,
            streaming=False,  # Non-streaming to enable shuffle and use with DataLoader
            split="train",
            block_size=args.block_size,
            logger=logger,
            accelerator=accelerator,
            seed=args.seed,
            use_tulu=args.use_tulu_dataset,
        )

    # Load or create validation dataset
    if args.validation_file:
        logger.info(f"Loading separate validation file from {args.validation_file}")
        # Load validation dataset from file (if you have a custom validation set)
        raw_eval_dataset = load_dataset(args.validation_file, split='validation', streaming=False)
        
        # Process validation dataset in the same way as training
        validation_dataset = load_sft_dataset(
            tokenizer=tokenizer,
            streaming=False,
            split="validation",
            block_size=args.block_size,
            logger=logger,
            accelerator=accelerator,
            seed=args.seed,
            use_tulu=False,  # Use the custom validation file
        )
    else:
        # Use a portion of the training set as validation
        logger.info("No validation file provided. Using a portion of training data for validation.")
        # Split dataset
        full_train_dataset = train_dataset["train"]
        train_val_split_point = int(len(full_train_dataset) * (100 - args.validation_split_percentage) / 100)
        
        # Create train/val split
        train_dataset = {"train": full_train_dataset.select(range(train_val_split_point))}
        validation_dataset = {"validation": full_train_dataset.select(range(train_val_split_point, len(full_train_dataset)))}
        
        logger.info(f"Split dataset: {len(train_dataset['train'])} training examples, {len(validation_dataset['validation'])} validation examples")

    # Create DataLoaders
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    train_dataloader = DataLoader(
        train_dataset["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    
    validation_dataloader = DataLoader(
        validation_dataset["validation"] if "validation" in validation_dataset else validation_dataset["train"],
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )
    
    #################
    # Prepare optimizer and scheduler
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, 
        lr=args.learning_rate, betas=[0.9, 0.95], weight_decay=args.weight_decay)
    
    # Scheduler and math around the number of training steps.
    prev_config = None
    prev_lr_scheduler_state = None

    # Check if we're resuming and need to adjust for GPU count changes
    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        dirs = [f.path for f in os.scandir(args.resume_from_checkpoint) if f.is_dir() and f.name.startswith("step_")]
        if len(dirs) > 0:
            dirs.sort(key=lambda x: int(os.path.basename(x).split("_")[1]))
            checkpoint_path = dirs[-1]
            config_path = os.path.join(checkpoint_path, "train_config.json")
            
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    prev_config = json.load(f)
                    logger.info(f"Loaded previous training configuration from {config_path}")
                    
                # Check if GPU count has changed
                if prev_config["num_processes"] != accelerator.num_processes:
                    logger.warning(
                        f"Number of GPUs changed from {prev_config['num_processes']} to {accelerator.num_processes}. "
                        f"Adjusting learning rate scheduler parameters to maintain training dynamics."
                    )

    # Initialize the scheduler based on previous config or current args
    if prev_config is not None:
        # Use the total_steps and warmup_steps from the previous run to maintain consistency
        # The scheduler will pick up from where it left off after loading the state
        total_steps = prev_config["max_train_steps"] * prev_config["num_processes"]
        warmup_steps = prev_config["num_warmup_steps"] * prev_config["num_processes"]
        
        logger.info(f"Initializing scheduler with previous parameters: "
                    f"total_steps={total_steps}, warmup_steps={warmup_steps}")
    else:
        # Calculate steps for new training
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
            logger.info(f"max_train_steps not provided, setting to {args.max_train_steps} based on epochs and dataloader size.")
        else:
            # Recalculate num_train_epochs if max_train_steps is provided
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
            logger.info(f"max_train_steps provided: {args.max_train_steps}. Setting num_train_epochs to {args.num_train_epochs}.")

        # Standard initialization for new training - Scale by number of processes
        total_steps = args.max_train_steps * accelerator.num_processes
        warmup_steps = args.num_warmup_steps * accelerator.num_processes
        logger.info(f"Initializing scheduler for new run: "
                    f"total_steps (scaled)={total_steps}, warmup_steps (scaled)={warmup_steps}, "
                    f"num_processes={accelerator.num_processes}")

    scheduler_specific_kwargs = {}
    if args.min_lr is not None:
        scheduler_specific_kwargs["min_lr"] = args.min_lr
        
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        scheduler_specific_kwargs=scheduler_specific_kwargs,
    )

    # Prepare everything with `accelerator`.
    if args.distillation:
        # 1. Prepare student model (default plugin is "student")
        logger.info("Preparing student model, optimizer, scheduler, and dataloaders with 'student' DeepSpeed config")
        # Ensure the correct plugin is active (should be by default)
        accelerator.state.select_deepspeed_plugin("student")
        model, optimizer, train_dataloader, validation_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, validation_dataloader, lr_scheduler
        )
        
        # 2. Prepare teacher model (switch plugin context)
        logger.info("Preparing teacher model with 'teacher' DeepSpeed config (inference)")
        accelerator.state.select_deepspeed_plugin("teacher")
        # Prepare only the teacher model; it doesn't need optimizer/scheduler
        # The dataloaders are already prepared under the student context, which is fine.
        teacher_model = accelerator.prepare(teacher_model)
        
        # Switch back to student plugin for the training loop if needed, 
        # although accumulate context should handle it.
        accelerator.state.select_deepspeed_plugin("student") 
        # logger.info(f"Active DeepSpeed plugin after preparations: {accelerator.state.deepspeed_plugin.config['zero_optimization']['stage']}") # Log stage of active plugin

    else:
        # Standard preparation if not using distillation
        logger.info("Preparing model, optimizer, scheduler, and dataloaders with single DeepSpeed config")
        model, optimizer, train_dataloader, validation_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, validation_dataloader, lr_scheduler
        )
    
    # Log model info after preparation
    alloc, max_alloc, reserved, max_reserved = get_memory_stats()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model parameters: {trainable_params/1024**3:.2f} B, device: {model.device}, dtype: {model.dtype}"
        f", Memory stats before training: Alloc: {alloc:.2f} G / {max_alloc:.2f} G, Resrv: {reserved:.2f} G / {max_reserved:.2f} G"
        , main_process_only=False)
    
    # Initialize trackers for tracking and logging
    if args.with_tracking:
        experiment_config = vars(args)
        accelerator.init_trackers("sft-training", experiment_config)

    # Resume training state if needed
    completed_steps = 0
    resume_step = None
    
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        dirs = [f.path for f in os.scandir(args.resume_from_checkpoint) if f.is_dir() and f.name.startswith("step_")]
        if len(dirs) > 0:
            dirs.sort(key=lambda x: int(os.path.basename(x).split("_")[1]))
            checkpoint_path = dirs[-1]
            path = os.path.basename(checkpoint_path)

            config_path = os.path.join(checkpoint_path, "train_config.json")
            
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    prev_config = json.load(f)
                    logger.info(f"Loaded previous training configuration from {config_path}")
                if "gradient_accumulation_steps" in prev_config:
                    original_grad_accum_steps = prev_config["gradient_accumulation_steps"]
                    logger.info(f"Original gradient accumulation steps: {original_grad_accum_steps}, current: {args.gradient_accumulation_steps}")

            # Load checkpoint state
            accelerator.load_state(os.path.join(checkpoint_path, "ckpt"))
            # Extract step number
            training_difference = os.path.splitext(path)[0]

            # Handle gradient accumulation steps difference
            if os.path.exists(config_path) and "gradient_accumulation_steps" in prev_config:
                resume_step = int(training_difference.replace("step_", "")) * original_grad_accum_steps
                completed_steps = resume_step // original_grad_accum_steps
                if original_grad_accum_steps != args.gradient_accumulation_steps:
                    # Convert resume_step to the new gradient accumulation steps
                    resume_step = (resume_step // original_grad_accum_steps) * args.gradient_accumulation_steps
                    
                    logger.warning(
                        f"Gradient accumulation steps changed from {original_grad_accum_steps} to {args.gradient_accumulation_steps}. "
                        f"Adjusted resume_step to {resume_step} to maintain consistent data position."
                    )
            else:
                resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
                completed_steps = resume_step // args.gradient_accumulation_steps
                
            logger.info(
                f"Resumed from checkpoint: {checkpoint_path}, resume steps (w. grad acc): {resume_step}, "
                f"completed_steps: {completed_steps}"
            )
            logger.info(f"Current learning rates after loading checkpoint: {lr_scheduler.get_last_lr()}")
        else:
            logger.warning(
                f"Please be aware that resume_from_checkpoint is specified as {args.resume_from_checkpoint}, "
                f"but no ckpt is detected"
            )
            # Initialize completed_steps to 0 if not resuming
            completed_steps = 0

    if not args.skip_train:
        #################
        # Train!
        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset['train'])}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        logger.info(f"  Steps per epoch = {num_update_steps_per_epoch}")

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process, initial=completed_steps)

        # Calculate starting epoch and step within epoch for resumption
        starting_epoch = 0
        resume_step_in_epoch = 0
        if args.resume_from_checkpoint and resume_step is not None:
            starting_epoch = completed_steps // num_update_steps_per_epoch
            # Calculate the number of *batches* processed within the starting epoch
            # `resume_step` tracks batches processed *before* gradient accumulation adjustment
            # `completed_steps` tracks optimizer steps.
            # We need the number of batches corresponding to the steps *within* the current epoch.
            steps_in_completed_epochs = starting_epoch * num_update_steps_per_epoch
            steps_this_epoch = completed_steps - steps_in_completed_epochs
            resume_step_in_epoch = steps_this_epoch * args.gradient_accumulation_steps # Convert optimizer steps back to batch steps for skipping
            logger.info(f"Resuming training from epoch {starting_epoch}, skipping {resume_step_in_epoch} batches in this epoch.")


        # Training loop
        model.train()
        if args.with_tracking:
            step_loss = torch.zeros(1, device=model.device, dtype=torch.float)

        for epoch in range(starting_epoch, args.num_train_epochs):
            logger.info(f"Starting Epoch {epoch+1}/{args.num_train_epochs}")
            active_dataloader = train_dataloader

            # Skip batches only in the first epoch of resumption
            if epoch == starting_epoch and resume_step_in_epoch > 0:
                logger.info(f"Skipping first {resume_step_in_epoch} batches for epoch {epoch+1} due to resumption.")
                active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step_in_epoch)
                # Reset skip marker after the first epoch
                resume_step_in_epoch = 0

            for step, batch in enumerate(active_dataloader):
                # Break loop if max_train_steps reached
                if completed_steps >= args.max_train_steps:
                    logger.info(f"Reached max_train_steps ({args.max_train_steps}). Stopping training.")
                    break

                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    # original_mean_loss is the mean loss calculated by the model (Eq 1 style for this micro-batch)
                    original_mean_loss = outputs.loss

                    # Calculate the number of valid (non-masked) tokens in the shifted labels for this micro-batch
                    # We need to shift the labels just like the model does internally for loss calculation
                    shift_labels = batch["labels"][..., 1:].contiguous()
                    num_valid_tokens = (shift_labels != -100).sum()

                    # If distillation is NOT used, the loss to scale is the original mean loss
                    loss_to_scale = original_mean_loss

                    # Add distillation loss if enabled
                    if args.distillation:
                        # Get teacher logits by running the teacher model on the same inputs
                        with torch.no_grad():
                            # Ensure teacher model is on the correct device if not prepared by accelerator
                            # teacher_model = teacher_model.to(accelerator.device)
                            teacher_outputs = teacher_model(**batch)
                            teacher_logits = teacher_outputs.logits # Shape: (batch_size, seq_len, vocab_size)

                        # Get student logits
                        student_logits = outputs.logits # Shape: (batch_size, seq_len, vocab_size)

                        # Shift logits and labels for distillation, matching the main CE loss calculation
                        shift_student_logits = student_logits[..., :-1, :].contiguous()
                        shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()
                        shift_labels = batch["labels"][..., 1:].contiguous()

                        # Create mask based on shifted labels
                        mask = shift_labels != -100 # Shape: (batch_size, seq_len - 1)

                        # Apply mask to the *shifted* logits
                        # Need to ensure mask is broadcastable or select elements correctly
                        masked_student_logits = shift_student_logits[mask] # Shape: (num_unmasked_tokens, vocab_size)
                        masked_teacher_logits = shift_teacher_logits[mask] # Shape: (num_unmasked_tokens, vocab_size)

                        # If there are tokens to compute loss on
                        if masked_student_logits.size(0) > 0:
                            # Compute distillation loss
                            distill_loss = compute_distillation_loss(
                                masked_student_logits,
                                masked_teacher_logits,
                                temperature=args.distillation_temperature
                            )

                            # Combine losses: (1 - alpha) * original_mean_loss + alpha * distillation_loss
                            # Note: We use the original_mean_loss here before scaling
                            combined_loss = (1 - args.distillation_alpha) * original_mean_loss + args.distillation_alpha * distill_loss

                            # If distillation IS used, the loss to scale is the combined loss
                            loss_to_scale = combined_loss

                    # Scale the loss (either original or combined) by the number of valid tokens
                    # to get the effective "sum" loss for this micro-batch
                    if num_valid_tokens > 0:
                        # scaled_loss approximates the result of using reduction='sum'
                        scaled_loss = loss_to_scale * num_valid_tokens.float()
                    else:
                        # Handle cases where a batch might have zero valid tokens (should be rare)
                        # scaled_loss = torch.tensor(0.0, device=loss_to_scale.device, dtype=loss_to_scale.dtype)
                        scaled_loss = loss_to_scale
                        logger.warning(f"rank {accelerator.process_index} at step {step} (epoch {epoch+1}) has no valid tokens, loss: {scaled_loss}", main_process_only=False)

                    # Use the scaled_loss for backpropagation
                    loss_for_backward = scaled_loss

                    if args.with_tracking:
                        # Log the original mean loss for easier interpretation/comparison,
                        # even though we backpropagate based on the scaled loss.
                        # If distilling, log the combined mean loss before scaling.
                        log_loss = loss_to_scale.detach().float() # Use loss_to_scale which is mean before scaling
                        step_loss += log_loss

                    # Perform backward pass using the scaled (sum) loss
                    accelerator.backward(loss_for_backward)

                    # Gradient Clipping (handled by DeepSpeedPlugin or manually if not using DS)
                    # if accelerator.sync_gradients and args.max_grad_norm is not None:
                    #    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks should be done outside the accumulate() context, after gradient sync
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1

                    if args.with_tracking:
                        # Average the logged *mean* loss across accumulation steps and devices
                        step_loss /= args.gradient_accumulation_steps
                        global_mean_loss = accelerator.reduce(step_loss, reduction='mean')

                        log_info = {"train_loss": global_mean_loss.item(), "epoch": epoch + 1} # Log the interpretable global mean loss

                        # Log distillation loss components if using distillation
                        if args.distillation:
                            log_info["distillation_alpha"] = args.distillation_alpha
                            log_info["distillation_temperature"] = args.distillation_temperature
                            # Log the mean CE and mean distill loss components if available
                            if 'distill_loss' in locals() and 'original_mean_loss' in locals():
                                # Ensure original_mean_loss is accessible or recalculate mean CE if needed
                                log_info["ce_loss"] = original_mean_loss.item()
                                log_info["distill_loss"] = distill_loss.item() # distill_loss is already a mean KL

                        for lr_idx, lr in enumerate(lr_scheduler.get_last_lr()):
                            log_info[f"lr_{lr_idx}"] = lr

                        accelerator.log(log_info, step=completed_steps)
                        step_loss.zero_()

                    # Checkpointing
                    if args.checkpointing_steps > 0 and completed_steps % args.checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)

                        # Save the training configuration
                        if accelerator.is_main_process:
                            train_config = {
                                "num_processes": accelerator.num_processes,
                                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                                "per_device_train_batch_size": args.per_device_train_batch_size,
                                "max_train_steps": args.max_train_steps,
                                "num_warmup_steps": args.num_warmup_steps,
                                "completed_steps": completed_steps, # Save the optimizer step count
                                "lr_scheduler_type": args.lr_scheduler_type,
                                "learning_rate": args.learning_rate,
                                "num_train_epochs": args.num_train_epochs, # Store epochs info
                            }
                            os.makedirs(output_dir, exist_ok=True)
                            with open(os.path.join(output_dir, "train_config.json"), "w") as f:
                                json.dump(train_config, f)

                        # Save checkpoint (Optimizer, LR Scheduler, RNG states)
                        # Use a specific sub-directory for accelerator state
                        accelerator_state_dir = os.path.join(output_dir, "accelerator_state")
                        # accelerator.save_state(accelerator_state_dir)

                        # Save model
                        model_dir = os.path.join(output_dir, "model")
                        # Ensure model saving directory exists if accelerator state saving didn't create the parent
                        os.makedirs(model_dir, exist_ok=True)
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(
                            model_dir, is_main_process=accelerator.is_main_process,
                            save_function=accelerator.save, state_dict=accelerator.get_state_dict(model)
                        )
                        if accelerator.is_main_process:
                            tokenizer.save_pretrained(model_dir)
                        logger.info(f"Model & checkpoint saved to {output_dir}", main_process_only=False)
                        accelerator.wait_for_everyone()

                    # Evaluation
                    if completed_steps > 0 and completed_steps % args.evaluate_every == 0:
                        losses = []
                        model.eval()
                        for eval_step, eval_batch in tqdm(
                            enumerate(validation_dataloader), desc=f"Eval epoch {epoch+1} step {completed_steps}",
                            total=len(validation_dataloader), disable=not accelerator.is_local_main_process
                        ):
                            with torch.no_grad():
                                outputs = model(**eval_batch)
                            # Gather losses across devices
                            # Repeat loss per item in batch before gathering
                            batch_size = eval_batch["input_ids"].shape[0]
                            loss = outputs.loss.repeat(batch_size)
                            gathered_losses = accelerator.gather_for_metrics(loss)
                            losses.append(gathered_losses)

                        # Calculate metrics on main process after gathering
                        if accelerator.is_main_process:
                            losses = torch.cat(losses)
                            # Truncate losses to actual dataset size if needed (dataloader might pad last batch)
                            # losses = losses[:len(validation_dataset["validation"])] # Adjust dataset split name if needed
                            try:
                                valid_loss = torch.mean(losses).item()
                                perplexity = math.exp(valid_loss)
                            except OverflowError:
                                perplexity = float('inf')
                                valid_loss = float('inf') # Ensure valid_loss is also set

                            logger.info(f"Epoch: {epoch+1}, Step: {completed_steps} | Perplexity: {perplexity:.2f} | Eval Loss: {valid_loss:.4f}")

                            if args.with_tracking:
                                accelerator.log(
                                    {
                                        "perplexity": perplexity,
                                        "eval_loss": valid_loss,
                                        "epoch": epoch + 1
                                    },
                                    step=completed_steps
                                )
                        # Sync to ensure all processes finish eval before continuing training
                        accelerator.wait_for_everyone()
                        model.train() # Switch back to train mode

                # Break inner loop if max steps reached
                if completed_steps >= args.max_train_steps:
                    break

            # Break outer loop if max steps reached
            if completed_steps >= args.max_train_steps:
                break

    # Final evaluation
    logger.info("***** Running final evaluation *****")
    losses = []
    model.eval()
    for step, batch in tqdm(
        enumerate(validation_dataloader), desc="Final Evaluation",
        disable=not accelerator.is_local_main_process
    ):
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss.repeat(args.per_device_eval_batch_size)
        losses.append(accelerator.gather(loss))

    # Calculate final metrics
    losses = torch.cat(losses)
    try:
        valid_loss = torch.mean(losses)
        perplexity = math.exp(valid_loss)
    except OverflowError:
        perplexity = float('inf')
    
    logger.info(f"Final results | Perplexity: {perplexity:.2f} | Eval Loss: {valid_loss:.4f}")
    
    if args.with_tracking:
        accelerator.log(
            {
                "perplexity": perplexity,
                "eval_loss": valid_loss,
            },
            step=completed_steps
        )

    # Report memory stats
    alloc, max_alloc, resv, max_resv = get_memory_stats()
    logger.info(
        f"Memory stats on exiting: Alloc: {alloc:.2f} G / {max_alloc:.2f} G, Resrv: {resv:.2f} G / {max_resv:.2f} G"
        , main_process_only=False)

    # Save final model
    if args.output_dir is not None:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, 
            save_function=accelerator.save, state_dict=accelerator.get_state_dict(model)
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            
            if args.push_to_hub:
                api = HfApi()
                # Ensure hub_model_id is set correctly
                repo_id = args.hub_model_id if args.hub_model_id else os.path.basename(args.output_dir)
                if not repo_id:
                    raise ValueError("Cannot push to hub without specifying --hub_model_id or a valid --output_dir name.")

                logger.info(f"Pushing final model to hub repository: {repo_id}")
                api.upload_folder(
                    commit_message="End of training",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )
                logger.info(f"Successfully pushed to {repo_id}")

                # Save final results (ensure metrics are calculated correctly before saving)
                # The calculation needs to happen after the final eval loop completes
                final_metrics = {}
                if 'perplexity' in locals() and 'valid_loss' in locals():
                     final_metrics = {"perplexity": perplexity.item() if isinstance(perplexity, torch.Tensor) else perplexity,
                                      "eval_loss": valid_loss.item() if isinstance(valid_loss, torch.Tensor) else valid_loss}
                     with open(os.path.join(args.output_dir, "final_results.json"), "w") as f:
                         json.dump(final_metrics, f)
                else:
                     logger.warning("Final perplexity/loss not available for saving in final_results.json. Was evaluation skipped or did it error?")


            logger.info(f"Saving final model to {args.output_dir} done!", main_process_only=False)
            accelerator.wait_for_everyone()
    
    if args.with_tracking:        
        accelerator.end_training()


if __name__ == "__main__":
    main() 