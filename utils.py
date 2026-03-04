import torch
import re
import json
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM,  BitsAndBytesConfig

def device_selection():
    if torch.cuda.is_available():
        return "cuda:0"
    elif torch.mps.is_available():
        return "mps"
    else:
        return "cpu"

def phishing_pipeline(device="cuda"):
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

    MODEL_ID = "google/gemma-3-4b-it"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map=device,
        torch_dtype=torch.bfloat16
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    model.config.use_cache = True
    model.eval()
    torch.set_grad_enabled(False)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        batch_size=1
    )
    return pipe


def phishing_pipeline_quantized(device="cuda"):
    """
    Create the pipeline for Llama 3.1 8B (quantized).
    """

    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

    MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        quantization_config=quant_config, 
        device_map=device)

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    
    # Set to evaluation mode
    model.config.use_cache = False 
    model.eval()
    torch.set_grad_enabled(False)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        batch_size=1 
    )
    return pipe

def extract_json_block(text):
    # Find the first {...} block including nested braces
    stack = []
    start = None

    for i, char in enumerate(text):
        if char == '{':
            if not stack:
                start = i
            stack.append(char)
        elif char == '}':
            stack.pop()
            if not stack and start is not None:
                return text[start:i+1]

    return None  # No valid JSON found