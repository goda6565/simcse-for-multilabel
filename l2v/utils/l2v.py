from peft import LoraConfig, get_peft_model, PeftModel  # type: ignore
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoConfig,
    AutoModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)


def setup_l2v() -> tuple[nn.Module, PreTrainedTokenizer | PreTrainedTokenizerFast]:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
    )
    config = AutoConfig.from_pretrained(
        "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp", trust_remote_code=True
    )
    # べースモデルの読み込み
    model = AutoModel.from_pretrained(
        "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
        trust_remote_code=True,
        config=config,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        attn_implementation="flash_attention_2",
        ignore_mismatched_sizes=True,
    )
    # MNTPのLoRA重みをベースモデルにマージ
    model = PeftModel.from_pretrained(
        model,
        "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
        ignore_mismatched_sizes=True,
    )
    model = model.merge_and_unload()

    # 教師なしモデルの読み込み
    model = PeftModel.from_pretrained(
        model, "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse",
        ignore_mismatched_sizes=True,
    )
    model = model.merge_and_unload()

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

    # LoRAを適用
    model = get_peft_model(model, config)  # type: ignore
    print("Model's Lora trainable parameters:")
    model.print_trainable_parameters()
    model = model.to("cuda:0")
    return model, tokenizer
