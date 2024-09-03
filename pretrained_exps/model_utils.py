import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

import torch

def get_tokenizer():
  return AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")


def get_model(model_name):
    
    if "mamba" in model_name:
        model = MambaLMHeadModel.from_pretrained(model_name,device="cuda",dtype=torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name,device_map="auto")

    return model
