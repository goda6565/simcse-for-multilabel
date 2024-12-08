import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)


def setup_bert() -> tuple[nn.Module, PreTrainedTokenizer | PreTrainedTokenizerFast]:
    base_model_name = "princeton-nlp/unsup-simcse-bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModel.from_pretrained(base_model_name)
    return model, tokenizer
