import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)


def setup_bert() -> tuple[nn.Module, PreTrainedTokenizer | PreTrainedTokenizerFast]:
    base_model_name = "google-bert/bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModel.from_pretrained(base_model_name)
    return model, tokenizer
