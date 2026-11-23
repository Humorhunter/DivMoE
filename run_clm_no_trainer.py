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
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer or Accelerator.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path
from contextlib import nullcontext
from tqdm.auto import tqdm

import datasets
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import deepspeed
from deepspeed.runtime.utils import set_random_seed, move_to_device
import deepspeed.comm as dist

from datasets import load_dataset, DatasetDict
from huggingface_hub import HfApi

# Import the OLMoE dataset loader
from prepare_data.olmoe_dataset import load_fixed_olmoe_mix_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
try:
    from transformers.utils import check_min_version
except ImportError:  # pragma: no cover - removed in newer transformers
    def check_min_version(*args, **kwargs):
        pass
try:
    from transformers.utils import send_example_telemetry
except ImportError:  # removed in newer transformers; unused by this script
    def send_example_telemetry(*args, **kwargs):
        pass
from transformers.utils.versions import require_version

# Import local DivMoE model for training
try:
    from modeling_divmoe import DivMoEConfig, DivMoEForCausalLM
except ImportError:
    pass  # DivMoE model is optional

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.50.0.dev0")

logger = logging.getLogger(__name__)

require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    # Add local_rank argument provided by deepspeed.launcher
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    
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
        "--olmoe_data_dir",
        type=str,
        default=None,
        help="Directory path containing OLMoE dataset files (.json.zst or .json.gz).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv, txt or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv, txt or a json file containing the validation data."
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
        required=False,
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
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "cosine_with_min_lr"],
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
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses for data loading. The OLMoE loader reads and "
             "tokenizes gzip/JSONL shards in Python; set >0 (e.g. 8) to parallelize across CPU cores.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
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
        help="Whether to enable experiment trackers for logging (TensorBoard by default).",
    )
    parser.add_argument(
        "--use_swanlab",
        action="store_true",
        help="Log metrics to SwanLab instead of TensorBoard (implies --with_tracking).",
    )
    parser.add_argument(
        "--swanlab_project",
        type=str,
        default="DivMoE",
        help="SwanLab project name.",
    )
    parser.add_argument(
        "--swanlab_run_name",
        type=str,
        default=None,
        help="SwanLab experiment name. Defaults to the output_dir basename.",
    )
    parser.add_argument(
        "--swanlab_workspace",
        type=str,
        default=None,
        help="SwanLab workspace/team name.",
    )
    parser.add_argument(
        "--swanlab_logdir",
        type=str,
        default=None,
        help="Local directory for SwanLab logs. Defaults to <output_dir>/swanlab.",
    )
    parser.add_argument(
        "--swanlab_mode",
        type=str,
        default=None,
        choices=["cloud", "local", "disabled", "offline"],
        help="SwanLab run mode (cloud/local/offline/disabled).",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="auto",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
        help="Attention backend. 'auto' uses flash_attention_2 if installed, else sdpa. "
             "'eager' is slow and memory-hungry; avoid on long sequences.",
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether to use gradient checkpointing to save memory at the expense of a slower backward pass.",
    )
    parser.add_argument(
        "--zero_stage",
        type=int,
        default=2,
        help="ZeRO optimization stage (0, 1, 2, or 3).",
    )
    parser.add_argument(
        "--cpu_offload_optimizer",
        action="store_true",
        help="Offload optimizer states to CPU to save GPU memory (requires ZeRO-2 or 3).",
    )
    parser.add_argument(
        "--cpu_offload_param",
        action="store_true",
        help="Offload model parameters to CPU to save GPU memory (requires ZeRO-3).",
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
        "--dataset_implementation", type=str, default="fixed",
        choices=["original", "improved", "fixed"],
        help="Which dataset implementation to use (original, improved, or fixed)."
    )
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None and args.olmoe_data_dir is None:
        raise ValueError("Need either a dataset name, a training/validation file, or an OLMoE data directory.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`train_file` should be a csv, json or txt file.")

    if args.max_train_steps is None:
        raise ValueError("--max_train_steps must be provided because the streaming OLMoE dataset has no known length.")

    if args.olmoe_data_dir is None:
        raise ValueError("--olmoe_data_dir is required; this training script only supports the OLMoE-mix loader.")

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
    teacher_logits = teacher_logits.detach().clone()  # Detach to prevent gradient flow to teacher

    # Apply temperature scaling to soften the distributions
    soft_student_logits = student_logits / temperature
    soft_teacher_logits = teacher_logits / temperature
    
    # Compute KL divergence loss
    loss_kl = F.kl_div(
        input=F.log_softmax(soft_student_logits, dim=-1),
        target=F.softmax(soft_teacher_logits, dim=-1),
        reduction='batchmean',
        log_target=False,
    )
    
    # Multiply by temperature^2 to scale gradients back up
    return loss_kl * (temperature ** 2)


def build_dataset(
    raw_dataset, tokenizer, block_size=4*1024, logger=None, is_main_process=True,
):
    #################
    # Preprocessing the datasets.
    # 1. Only load text fields for the dataloader
    column_names = raw_dataset.column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        max_len = block_size if block_size is not None else tokenizer.model_max_length
        max_len = min(max_len, tokenizer.model_max_length)
        return tokenizer(examples[text_column_name], truncation=True, max_length=max_len)

    if is_main_process:
        tokenized_dataset = raw_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
        )
    else:
        tokenized_dataset = raw_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
        )

    # 2. Padding to max length    
    if block_size is None:
        block_size = tokenizer.model_max_length
    else:
        if block_size > tokenizer.model_max_length and logger is not None:
            logger.warning(
                f"The block_size passed ({block_size}) is larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
    )

    return lm_dataset


class SwanLabScalarWriter:
    """Adapter exposing the SummaryWriter.add_scalar/close interface backed by SwanLab."""

    def __init__(self, project, run_name=None, workspace=None, logdir=None, config=None, mode=None):
        import swanlab

        self._swanlab = swanlab
        init_kwargs = {"project": project, "config": config or {}}
        if run_name:
            init_kwargs["experiment_name"] = run_name
        if workspace:
            init_kwargs["workspace"] = workspace
        if mode:
            init_kwargs["mode"] = mode
        if logdir:
            os.makedirs(logdir, exist_ok=True)
            init_kwargs["logdir"] = logdir
            os.environ.setdefault("SWANLAB_SAVE_DIR", logdir)
        try:
            swanlab.init(**init_kwargs)
        except TypeError:
            init_kwargs.pop("logdir", None)
            swanlab.init(**init_kwargs)

    def add_scalar(self, tag, value, step):
        if hasattr(value, "item"):
            value = value.item()
        self._swanlab.log({tag: value}, step=step)

    def close(self):
        self._swanlab.finish()


def _resolve_attn_impl(choice: str) -> str:
    """Resolve 'auto' to flash_attention_2 when available, otherwise sdpa."""
    if choice != "auto":
        return choice
    try:
        from transformers.utils import is_flash_attn_2_available
        if is_flash_attn_2_available():
            return "flash_attention_2"
    except Exception:  # noqa: BLE001
        pass
    return "sdpa"


def create_deepspeed_config(args):
    """Create DeepSpeed configuration"""
    # Calculate cosine minimum ratio if min_lr is set
    cos_min_ratio = 0.0
    if args.min_lr is not None and args.learning_rate > 0:
        cos_min_ratio = args.min_lr / args.learning_rate

    ds_config = {
        "train_micro_batch_size_per_gpu": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "optimizer": { 
            "type": "AdamW", 
            "params": { 
                "lr": args.learning_rate, 
                "betas": [0.9, 0.95],
                "weight_decay": args.weight_decay
            } 
        },
        "scheduler": {
            "type": "WarmupCosineLR" if args.lr_scheduler_type == "cosine_with_min_lr" else "WarmupLR",
            "params": {
                "total_num_steps": args.max_train_steps,
                "warmup_num_steps": args.num_warmup_steps,
                "warmup_min_ratio": 0.0,
                "cos_min_ratio": cos_min_ratio,
            }
        },
        "fp16": { "enabled": False },
        "bf16": { "enabled": True },
        "gradient_clipping": 1.0,
    }
    
    # Configure ZeRO optimization
    zero_config = {
        "stage": args.zero_stage,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": True,
    }
    
    # Add CPU offloading configurations
    if args.cpu_offload_optimizer:
        logger.info("Enabling CPU offloading for optimizer states")
        zero_config["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True,
        }
    
    if args.cpu_offload_param:
        logger.info("Enabling CPU offloading for model parameters")
        zero_config["offload_param"] = {
            "device": "cpu",
            "pin_memory": True,
        }
    
    ds_config["zero_optimization"] = zero_config

    # Add gradient checkpointing to config if enabled
    if args.gradient_checkpointing:
        ds_config["gradient_checkpointing"] = { "enabled": True }
    return ds_config


def save_model_and_tokenizer(model, tokenizer, save_dir, is_main_process, args, completed_steps=None):
    """Save model and tokenizer with proper configuration"""
    if not is_main_process:
        return
    
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Saving model and tokenizer to {save_dir}")
    
    # Save the actual model (unwrapped from DeepSpeed)
    if hasattr(model, 'module'):
        actual_model = model.module
    else:
        actual_model = model
    
    # Save model with proper configuration
    actual_model.save_pretrained(
        save_dir,
        safe_serialization=True,
        save_function=torch.save
    )
    
    # Save tokenizer
    tokenizer.save_pretrained(save_dir)
    
    # Save training arguments for reproducibility
    training_args = {
        'model_name_or_path': args.model_name_or_path,
        'use_divmoe': args.use_divmoe,
        'block_size': args.block_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'seed': args.seed,
        'gradient_checkpointing': args.gradient_checkpointing,
        'zero_stage': args.zero_stage,
        'max_train_steps': args.max_train_steps,
        'completed_steps': completed_steps,
        'distillation': args.distillation,
        'teacher_model_name_or_path': args.teacher_model_name_or_path,
        'distillation_alpha': args.distillation_alpha,
        'distillation_temperature': args.distillation_temperature,
    }
    
    training_args_path = os.path.join(save_dir, "training_args.json")
    with open(training_args_path, "w") as f:
        json.dump(training_args, f, indent=2)
    
    logger.info(f"✅ Successfully saved model and tokenizer to {save_dir}")


def main():
    args = parse_args()

    # === Distributed setup ===
    deepspeed.init_distributed(dist_backend="nccl")
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    set_random_seed(args.seed)
    world_size = dist.get_world_size()

    is_main_process = dist.get_rank() == 0
    if is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    # === Logging setup ===
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process else logging.WARNING,
    )
    logger.info(f"Starting script on rank {dist.get_rank()}/{world_size}")
    
    # Set up datasets logging
    if is_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Setup experiment tracker (SwanLab or TensorBoard)
    writer = None
    if is_main_process:
        if args.use_swanlab:
            args.with_tracking = True
            run_name = args.swanlab_run_name or (os.path.basename(os.path.normpath(args.output_dir)) if args.output_dir else None)
            logdir = args.swanlab_logdir or (os.path.join(args.output_dir, "swanlab") if args.output_dir else None)
            writer = SwanLabScalarWriter(
                project=args.swanlab_project,
                run_name=run_name,
                workspace=args.swanlab_workspace,
                logdir=logdir,
                config=vars(args),
                mode=args.swanlab_mode,
            )
            logger.info(f"SwanLab tracking enabled: project={args.swanlab_project}, run={run_name}")
        elif args.with_tracking:
            writer = SummaryWriter(args.output_dir)

    # === Model and Tokenizer ===
    attn_impl = _resolve_attn_impl(args.attn_implementation)
    logger.info(f"Using attention implementation: {attn_impl}")
    if args.use_divmoe:
        # Check if DivMoE is available
        if 'DivMoEForCausalLM' not in globals():
            raise ImportError(
                "DivMoE model not found. Make sure modeling_divmoe.py is in the same directory."
            )
        
        # Initialize DivMoE model from original model path
        logger.info(f"Initializing DivMoE model from {args.model_name_or_path}")
        model = DivMoEForCausalLM.from_pretrained(
            args.model_name_or_path, 
            attn_implementation=attn_impl,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,  # Enable this
            trust_remote_code=args.trust_remote_code,
        )
        logger.info(f"DivMoE model initialized from {args.model_name_or_path}")
    elif args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            low_cpu_mem_usage=True,  # Enable this
            trust_remote_code=args.trust_remote_code,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )
    else:
        raise ValueError(
            "You need to specify a `model_name_or_path` if you are not using DivMoE. "
            "This should be a valid model identifier from the Hugging Face Hub or a local path."
        )
        
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name or args.model_name_or_path, 
        use_fast=not args.use_slow_tokenizer, 
        trust_remote_code=args.trust_remote_code
    )

    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        logger.info("Enabling gradient checkpointing")
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            if hasattr(model.config, 'use_cache'):
                model.config.use_cache = False
                logger.info("Disabled use_cache for gradient checkpointing")

    # Move model to device
    model = model.to(device)

    alloc, max_alloc, reserved, max_reserved = get_memory_stats()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model parameters: {trainable_params/1024**3:.2f} B, device: {model.device}, dtype: {model.dtype}"
        f", Memory stats after initializing model: Alloc: {alloc:.2f} G / {max_alloc:.2f} G, Resrv: {reserved:.2f} G / {max_reserved:.2f} G"
    )
    logger.info(
        f"Model emb size: {model.get_input_embeddings().weight.shape[0]}"
        f", tokenizer vocab size: {len(tokenizer)}"
    )

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
        # teacher_model = teacher_model.to(device)
        
        # Count teacher parameters
        teacher_params = sum(p.numel() for p in teacher_model.parameters())
        logger.info(
            f"Teacher model loaded with {teacher_params/1024**3:.2f} B parameters, {trainable_params/1024**3:.2f} B student parameters"
        )
        
        # Memory stats after loading teacher
        alloc, max_alloc, reserved, max_reserved = get_memory_stats()
        logger.info(
            f"Memory stats after loading teacher model: Alloc: {alloc:.2f} G / {max_alloc:.2f} G, Resrv: {reserved:.2f} G / {max_reserved:.2f} G"
        )

    # We resize the embeddings only when necessary to avoid index errors.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    max_embedding_size = max(embedding_size, len(tokenizer))
    if teacher_model:
        max_embedding_size = max(max_embedding_size, teacher_model.get_input_embeddings().weight.shape[0])
    if max_embedding_size > embedding_size:
        model.resize_token_embeddings(max_embedding_size)
        logger.info(f"Resized token embeddings from {embedding_size} to {max_embedding_size}")

    

    # === DeepSpeed Initialization ===
    ds_config = create_deepspeed_config(args)
    
    # Log configuration
    if is_main_process:
        logger.info("=== DeepSpeed Configuration ===")
        logger.info(json.dumps(ds_config, indent=2))

    # Initialize student model with DeepSpeed
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        config=ds_config,
        dist_init_required=False,  # Already initialized above
    )

    # Initialize teacher model with DeepSpeed for distillation (inference only)
    if args.distillation:
        teacher_ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            "bf16": {"enabled": True},
            "zero_optimization": {
                "stage": 3,
                "overlap_comm": True,
                "offload_param": {"device": "cpu", "pin_memory": True},
            },
        }
        teacher_model, _, _, _ = deepspeed.initialize(
            model=teacher_model,
            config=teacher_ds_config,
            dist_init_required=False,
        )

    alloc, max_alloc, reserved, max_reserved = get_memory_stats()
    logger.info(
        f"Memory stats after DeepSpeed initialization: Alloc: {alloc:.2f} G / {max_alloc:.2f} G, Resrv: {reserved:.2f} G / {max_reserved:.2f} G"
    )


    # === Checkpoint Loading ===
    completed_steps = 0
    if args.resume_from_checkpoint:
        _, client_state = model.load_checkpoint(args.resume_from_checkpoint)
        if client_state is not None and 'step' in client_state:
            completed_steps = client_state['step']
            logger.info(f"Resumed from step {completed_steps} from checkpoint {args.resume_from_checkpoint}")
        else:
            logger.warning(f"Could not find step in client state from checkpoint {args.resume_from_checkpoint}")

    # === Datasets ===
    logger.info(f"Loading OLMoE dataset from {args.olmoe_data_dir} using {args.dataset_implementation} implementation")
    
    if args.dataset_implementation != "fixed":
        logger.warning(
            f"Dataset implementation '{args.dataset_implementation}' is not included in this release; "
            "falling back to the fixed OLMoE-mix loader."
        )
    raw_train_dataset = load_fixed_olmoe_mix_dataset(
        args.olmoe_data_dir,
        seed=args.seed,
        global_step=completed_steps,
        buffer_size=100_000,
    )

    train_dataset = build_dataset(raw_train_dataset['train'], tokenizer, block_size=args.block_size, logger=logger, is_main_process=is_main_process)
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=default_data_collator,
        batch_size=args.per_device_train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    validation_dataloader = None
    if args.validation_file is not None:
        logger.info(f"Loading separate validation file from {args.validation_file}")
        raw_eval_dataset = load_dataset(args.validation_file, split='validation', streaming=True)
        validation_dataset = build_dataset(raw_eval_dataset, tokenizer, block_size=args.block_size, logger=logger, is_main_process=is_main_process)
        validation_dataloader = DataLoader(
            validation_dataset,
            collate_fn=default_data_collator,
            batch_size=args.per_device_eval_batch_size,
            num_workers=args.dataloader_num_workers,
        )

    from accelerate import Accelerator
    accelerator = Accelerator()
    train_dataloader = accelerator.prepare(train_dataloader)
    if validation_dataloader is not None:
        validation_dataloader = accelerator.prepare(validation_dataloader)

    # Log first few batches for debugging
    for idx, batch in enumerate(train_dataloader):
        logger.warning(f"rank {dist.get_rank()} batch {idx}: {batch['input_ids'][0, :5].tolist()}")
        if idx == 2:
            break

    # === Training Loop ===
    if not args.skip_train:
        logger.info("***** Running training *****")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {args.per_device_train_batch_size * world_size * args.gradient_accumulation_steps}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")

        progress_bar = tqdm(range(args.max_train_steps), disable=not is_main_process)
        progress_bar.update(completed_steps)

        model.train()
        if args.with_tracking:
            step_loss = torch.zeros(1, device=model.device, dtype=torch.float)

        for step, batch in enumerate(train_dataloader):
            if completed_steps >= args.max_train_steps:
                break
                
            batch = move_to_device(batch, device)
            outputs = model(**batch)
            loss = outputs.loss
            original_clm_loss = loss  # Keep the original CLM loss for logging

            # Add distillation loss if enabled
            if args.distillation:
                student_logits = outputs.logits
                
                with torch.no_grad():
                    teacher_outputs = teacher_model(**batch)
                    teacher_logits = teacher_outputs.logits
                
                # Apply shifting for CLM
                shift_student_logits = student_logits[..., :-1, :].contiguous()
                shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()
                shift_labels = batch["labels"][..., 1:].contiguous()

                # Create mask based on shifted labels
                mask = shift_labels != -100
                
                # Apply mask to the shifted logits
                masked_student_logits = shift_student_logits[mask]
                masked_teacher_logits = shift_teacher_logits[mask]

                if masked_student_logits.size(0) > 0:
                    distill_loss = compute_distillation_loss(
                        masked_student_logits,
                        masked_teacher_logits,
                        temperature=args.distillation_temperature
                    )
                    # Combine losses
                    loss = (1 - args.distillation_alpha) * original_clm_loss + args.distillation_alpha * distill_loss
                else:
                    distill_loss = torch.tensor(0.0, device=loss.device)
                    loss = original_clm_loss

            # DeepSpeed backward and step
            loss = model.backward(loss)
            model.step()

            if args.with_tracking:
                step_loss += loss.detach().float()

            # Check if we've completed a full optimization step
            if model.is_gradient_accumulation_boundary():
                progress_bar.update(1)
                completed_steps += 1

                if args.with_tracking and is_main_process:
                    current_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler else optimizer.param_groups[0]['lr']
                    logger.info(f"completed_steps {completed_steps}: loss: {step_loss.item():.4f}, lr: {current_lr:.2e}")
                    
                    log_info = {"train_loss": step_loss.item()}
                    
                    # Log distillation loss components if using distillation
                    if args.distillation:
                        log_info["distillation_alpha"] = args.distillation_alpha
                        log_info["distillation_temperature"] = args.distillation_temperature
                        if 'distill_loss' in locals() and 'original_clm_loss' in locals():
                            log_info["clm_loss_component"] = original_clm_loss.item()
                            log_info["distill_loss_component"] = distill_loss.item()
                    
                    log_info["learning_rate"] = current_lr
                    
                    if writer:
                        for key, value in log_info.items():
                            writer.add_scalar(f"train/{key}", value, completed_steps)
                    
                    step_loss.zero_()

                if args.checkpointing_steps > 0 and completed_steps % args.checkpointing_steps == 0:
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{completed_steps}")
                    model.save_checkpoint(checkpoint_dir, client_state={'step': completed_steps})
                    
                    # Also save model checkpoint
                    model_dir = os.path.join(args.output_dir, f"model-{completed_steps}")
                    save_model_and_tokenizer(model, tokenizer, model_dir, is_main_process, args, completed_steps)

                if validation_dataloader is not None and completed_steps % args.evaluate_every == 0:
                    evaluate(model, validation_dataloader, completed_steps, writer, is_main_process, teacher_model, args)
                    model.train()

    # === Final Evaluation ===
    if validation_dataloader is not None:
        logger.info("***** Running Final Evaluation *****")
        evaluate(model, validation_dataloader, completed_steps, writer, is_main_process, teacher_model, args)

    # === Save Final Model ===
    alloc, max_alloc, reserved, max_reserved = get_memory_stats()
    logger.info(
        f"Memory stats on exiting: Alloc: {alloc:.2f} G / {max_alloc:.2f} G, Resrv: {reserved:.2f} G / {max_reserved:.2f} G"
    )

    if args.output_dir is not None:
        save_model_and_tokenizer(model, tokenizer, args.output_dir, is_main_process, args, completed_steps)
        logger.info(f"Saving model to {args.output_dir} done!")

    if writer:
        writer.close()
    
    logger.info("Training finished.")


def evaluate(model, validation_dataloader, completed_steps, writer, is_main_process, teacher_model=None, args=None):
    """Evaluate the model on validation set"""
    model.eval()
    losses = []
    
    eval_iterator = tqdm(validation_dataloader, desc=f"Evaluation at step {completed_steps}", disable=not is_main_process)
    
    for batch in eval_iterator:
        batch = move_to_device(batch, model.device)
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            
            # Gather losses from all ranks
            gathered_losses = [torch.zeros_like(loss) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_losses, loss)
            losses.extend(gathered_losses)

    if is_main_process:
        if not losses:
            logger.warning(f"Evaluation at step {completed_steps} resulted in no losses, skipping perplexity calculation.")
            return

        try:
            eval_loss = torch.mean(torch.stack(losses))
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float('inf')
        
        logger.info(f"step: {completed_steps} perplexity: {perplexity:.2f} eval_loss: {eval_loss.item():.4f}")
        
        if writer:
            writer.add_scalar("eval/perplexity", perplexity, completed_steps)
            writer.add_scalar("eval/loss", eval_loss.item(), completed_steps)

if __name__ == "__main__":
    main()