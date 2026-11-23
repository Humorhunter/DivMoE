#!/usr/bin/env python
# coding=utf-8
"""
Script to create a DivMoE model from domain-specialized models and save it to a specified path.
This script initializes the DivMoE model, its tokenizer and vocabulary.
"""

import argparse
import logging
import os
import torch
from typing import List

from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer
from modeling_divmoe import DivMoEForCausalLM, DivMoEConfig

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Create a DivMoE model from domain-specialized models")
    
    # Model paths
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="Path to the base pre-trained model (e.g., Qwen2.5-3B)"
    )
    
    parser.add_argument(
        "--domain_model_paths",
        type=str,
        nargs="+",
        required=True,
        help="Paths to the domain-specialized models"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the DivMoE model, tokenizer, and vocabulary"
    )
    
    # Model configuration
    parser.add_argument(
        "--experts_per_domain",
        type=int,
        default=4,
        help="Number of experts to create from each domain model"
    )
    
    parser.add_argument(
        "--num_experts_per_token",
        type=int,
        default=2,
        help="Number of experts to route each token to"
    )
    
    parser.add_argument(
        "--aux_loss_alpha",
        type=float,
        default=0.01,
        help="Weight for the auxiliary load balancing loss"
    )
    
    parser.add_argument(
        "--use_shared_expert",
        action="store_true",
        help="Whether to use a shared generalist expert"
    )
    
    parser.add_argument(
        "--norm_router_scores",
        action="store_true",
        default=True,
        help="Whether to normalize router scores"
    )
    
    parser.add_argument(
        "--router_bias",
        action="store_true",
        help="Whether to use bias in router"
    )
    
    parser.add_argument(
        "--half_precision",
        action="store_true",
        help="Save model in half precision (fp16)"
    )
    
    parser.add_argument(
        "--bf16_precision",
        action="store_true",
        help="Save model in bfloat16 precision (bf16)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()

def setup_logging(verbose: bool):
    logging_level = logging.INFO
    if verbose:
        logging_level = logging.DEBUG
        
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging_level,
    )

def create_and_save_divmoe(args):
    """Create a DivMoE model from domain-specialized models and save it."""
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer from base model
    logger.info(f"Loading tokenizer from {args.base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    
    # Log model configuration
    logger.info(f"Creating DivMoE with:")
    logger.info(f"  Base model: {args.base_model_path}")
    logger.info(f"  Domain models: {args.domain_model_paths}")
    logger.info(f"  Number of domain groups: {len(args.domain_model_paths)}")
    logger.info(f"  Experts per domain: {args.experts_per_domain}")
    logger.info(f"  Total experts: {len(args.domain_model_paths) * args.experts_per_domain}")
    logger.info(f"  Experts per token: {args.num_experts_per_token}")
    logger.info(f"  Using shared expert: {args.use_shared_expert}")
    
    # Create DivMoE model
    logger.info("Initializing DivMoE model...")
    logger.info(f"Each domain specialist expert will have 1/{args.experts_per_domain} of the original MLP intermediate size")
    model = DivMoEForCausalLM.from_domain_models(
        base_model_name_or_path=args.base_model_path,
        domain_model_name_or_paths=args.domain_model_paths,
        num_experts_per_domain=args.experts_per_domain,
        num_experts_per_tok=args.num_experts_per_token,
        aux_loss_alpha=args.aux_loss_alpha,
        use_shared_expert=args.use_shared_expert,
        norm_router_scores=args.norm_router_scores,
        router_bias=args.router_bias,
    )
    
    # Convert to half precision or bf16 if requested
    if args.half_precision and args.bf16_precision:
        logger.warning("Both --half_precision and --bf16_precision specified. Using bf16 precision.")
        args.half_precision = False

    if args.half_precision:
        logger.info("Converting model to half precision (fp16)...")
        model = model.half()
    elif args.bf16_precision:
        logger.info("Converting model to bfloat16 precision (bf16)...")
        model = model.to(torch.bfloat16)
    
    # Save the model
    logger.info(f"Saving model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    
    # Save the tokenizer
    logger.info(f"Saving tokenizer to {args.output_dir}")
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("DivMoE model creation complete!")
    logger.info(f"Model and tokenizer saved to {args.output_dir}")

def main():
    args = parse_args()
    setup_logging(args.verbose)
    
    try:
        create_and_save_divmoe(args)
    except Exception as e:
        logger.exception(f"Error creating DivMoE model: {e}")
        raise

if __name__ == "__main__":
    main() 