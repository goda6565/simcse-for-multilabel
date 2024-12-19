import torch
from torch import Tensor
from transformers import BatchEncoding, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"


def eval_collate_fn(
    examples: list[dict],
    max_length: int = 128,
) -> dict[str, BatchEncoding | Tensor]:
    """SimCSEの検証・テストセットのミニバッチを作成"""
    # トークナイザを適用する
    tokenized_texts = tokenizer(
        [example["text"] for example in examples],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )  # type: ignore

    # データセットに付与されたラベル配列のTensorを作成する
    label = torch.tensor([example["labels"] for example in examples])

    return {
        "tokenized_text": tokenized_texts,
        "labels": label,
    }


def sup_scl_train_collate_fn(
    examples: list[dict],
    max_length: int = 128,
) -> dict[str, BatchEncoding | list[Tensor]]:
    """訓練セットのミニバッチを作成"""
    same_label_index = []
    for example in examples:
        index = []
        for i, pair in enumerate(examples):
            if example["labels"] == pair["labels"]:
                index.append(i)
        same_label_index.append(index)

    # ミニバッチに含まれる前提文と仮説文にトークナイザを適用する
    tokenized_texts1 = tokenizer(
        [example["text"] for example in examples],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    tokenized_texts2 = tokenizer(
        [example["same_label_text"] for example in examples],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    labels = []
    for i in range(len(examples)):
        labels.append(torch.tensor(same_label_index[i]))

    return {
        "tokenized_texts_1": tokenized_texts1,
        "tokenized_texts_2": tokenized_texts2,
        "labels": labels,
    }


def sup_not_scl_train_collate_fn(
    examples: list[dict],
    max_length: int = 128,
) -> dict[str, BatchEncoding | list[Tensor]]:
    """訓練セットのミニバッチを作成"""
    same_label_index = []
    for example in examples:
        index = []
        for i, pair in enumerate(examples):
            # ラベルのリスト同士の共通部分を確認
            if set(example["labels"]) & set(pair["labels"]):
                index.append(i)
        same_label_index.append(index)

    # ミニバッチ��含まれる前提文と仮説文にトークナイザを適用する
    tokenized_texts1 = tokenizer(
        [example["text"] for example in examples],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    tokenized_texts2 = tokenizer(
        [example["same_label_text"] for example in examples],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    labels = []
    for i in range(len(examples)):
        labels.append(torch.tensor(same_label_index[i]))

    return {
        "tokenized_texts_1": tokenized_texts1,
        "tokenized_texts_2": tokenized_texts2,
        "label_list": [example["labels"] for example in examples],
        "labels": labels,
    }
