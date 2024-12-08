import torch
import torch.nn as nn
from peft import PeftModel  # type: ignore
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    AutoConfig,
)


def load_bert(
    model_path: str,
) -> tuple[nn.Module, PreTrainedTokenizer | PreTrainedTokenizerFast]:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    encoder = AutoModel.from_pretrained(model_path)
    return encoder, tokenizer


def load_l2v(
    model_path: str,
) -> tuple[nn.Module, PreTrainedTokenizer | PreTrainedTokenizerFast]:
    tokenizer = AutoTokenizer.from_pretrained(
        "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
    )
    config = AutoConfig.from_pretrained(
        "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp", trust_remote_code=True
    )
    # べースモデルの読み込み
    decoder = AutoModel.from_pretrained(
        "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
        trust_remote_code=True,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    # MNTPのLoRA重みをベースモデルにマージ
    decoder = PeftModel.from_pretrained(
        decoder,
        "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
    )
    decoder = decoder.merge_and_unload()

    # 教師ありモデルの重みをマージ
    decoder = PeftModel.from_pretrained(
        decoder,
        "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse",
    )
    decoder = decoder.merge_and_unload()
    # 今回の実験で学習したモデルの重みをマージ
    decoder = PeftModel.from_pretrained(
        decoder,
        model_path,
    )
    decoder = decoder.merge_and_unload()
    return decoder, tokenizer
