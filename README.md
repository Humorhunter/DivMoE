# DivMoE: Supplementary Code

This repository contains the core implementation of DivMoE (Diversity-augmented Mixture of Experts), a framework for upcycling dense language models into fine-grained MoE architectures.

## File Structure

```
supplementary_code/
├── README.md                       # This file
├── modeling_divmoe.py              # Core DivMoE model implementation
├── create_divmoe.py                # Script to create DivMoE from domain specialists
├── run_clm_no_trainer.py           # Continuous pre-training script
├── run_sft_no_trainer.py           # Supervised fine-tuning script
├── utils.py                        # Utility functions
└── data/
    ├── olmoe_dataset.py            # OLMoE-mix dataset loader for pre-training
    ├── sft_dataset.py              # SFT dataset processing utilities
    ├── nemotron_sft_dataset.py     # Nemotron-style SFT dataset with sequence packing
    └── tulu_dataset.py             # Tulu dataset loader
```

## Key Components

### 1. `modeling_divmoe.py`
Core implementation of the DivMoE architecture including:
- `DivMoEConfig`: Configuration class for DivMoE models
- `DivMoERouter`: Diversity-constrained routing mechanism
- `DivMoEMLP`: Mixture of Experts MLP layer with domain-specialized experts
- `DivMoEForCausalLM`: Full model with language modeling head
- `from_domain_models()`: Class method to initialize from domain specialists via vertical FFN slicing

### 2. `create_divmoe.py`
Command-line script to create a DivMoE model from:
- A base pre-trained model
- Multiple domain-specialized models (fine-tuned on math, code, science, etc.)

Usage:
```bash
python create_divmoe.py \
    --base_model_path /path/to/base_model \
    --domain_model_paths /path/to/math_model /path/to/code_model /path/to/science_model /path/to/commonsense_model \
    --output_dir /path/to/output \
    --experts_per_domain 2 \
    --num_experts_per_token 2 \
    --use_shared_expert
```

### 3. `run_clm_no_trainer.py`
Continuous pre-training script with:
- DeepSpeed ZeRO integration for distributed training
- OLMoE-mix dataset streaming
- Gradient checkpointing and CPU offloading support
- Optional knowledge distillation from a teacher model

### 4. `run_sft_no_trainer.py`
Supervised fine-tuning script with:
- HuggingFace Accelerate integration
- Proper loss masking on assistant responses only
- Multi-turn conversation support
- Optional knowledge distillation

### 5. `utils.py`
Utility functions for:
- GPU memory statistics
- Model registration with HuggingFace Auto classes
- Parameter counting and expert statistics

### 6. `data/`
Dataset utilities for training:
- `olmoe_dataset.py`: OLMoE-mix dataset loader for continuous pre-training
- `sft_dataset.py`: Basic SFT dataset with conversation tokenization and loss masking
- `nemotron_sft_dataset.py`: Packed SFT dataset with sequence packing for efficient training
- `tulu_dataset.py`: Tulu-3 dataset loader

## Requirements

- Python >= 3.8
- PyTorch >= 2.0
- transformers >= 4.40.0
- datasets
- flash-attn (optional, for Flash Attention 2 support)

## Model Architecture

DivMoE converts each FFN layer in a dense transformer into a Mixture of Experts layer:

1. **Domain Specialists**: The base model is fine-tuned into n domain specialists
2. **Expert Slicing**: Each specialist's FFN is vertically sliced into m fine-grained experts
3. **Diversity-Constrained Routing**: The router selects k experts from different domain groups

Key hyperparameters:
- `num_domain_groups` (n): Number of domain groups (default: 4)
- `experts_per_domain` (m): Experts per domain group (default: 2)
- `num_experts_per_tok` (k): Active experts per token (default: 2)

## License

This code is released for academic research purposes.
