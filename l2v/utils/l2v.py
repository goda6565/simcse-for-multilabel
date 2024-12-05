from llm2vec import LLM2Vec
from peft import LoraConfig, get_peft_model  # type: ignore
import torch
import torch.nn as nn
from transformers import AutoTokenizer, BitsAndBytesConfig


def setup_l2v() -> tuple[nn.Module, AutoTokenizer]:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = LLM2Vec.from_pretrained(
        base_model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct",
        enable_bidirectional=True,
        peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
        merge_peft=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        attention_dropout=0.3,
        quantization_config=quantization_config,
        use_cache=False,
        device_map="auto",
    )

    lora_modules = [
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=lora_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=None,
    )

    model.model = get_peft_model(model.model, config)  # type: ignore
    print("Model's Lora trainable parameters:")
    model.model.print_trainable_parameters()
    tokenizer = model.tokenizer
    model = model.model.to("cuda:0")
    return model, tokenizer
