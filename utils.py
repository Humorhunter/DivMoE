import torch
from transformers import AutoConfig, GenerationConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM

# Importing modeling_divmoe will auto-register the model
from modeling_divmoe import DivMoEConfig, DivMoEModel, DivMoEForCausalLM

GB = 1024**3
def get_memory_stats():
    alloc = torch.cuda.memory_allocated() / GB
    max_alloc = torch.cuda.max_memory_allocated() / GB
    reserved = torch.cuda.memory_reserved() / GB
    max_reserved = torch.cuda.max_memory_reserved() / GB
    return alloc, max_alloc, reserved, max_reserved

def register_custom_model():
    """
    Register DivMoE model with transformers Auto classes.
    Note: This is now handled automatically when modeling_divmoe is imported,
    but this function is kept for backward compatibility.
    """
    try:
        AutoConfig.register("divmoe", DivMoEConfig)
        AutoModel.register(DivMoEConfig, DivMoEModel)
        AutoModelForCausalLM.register(DivMoEConfig, DivMoEForCausalLM)
    except ValueError:
        # Model already registered (either by auto-registration or previous call)
        pass