import torch
from torch import Tensor
from transformers import BatchEncoding, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"


def eval_collate_fn(
    examples: list[dict],
) -> dict[str, BatchEncoding | Tensor]:
    """SimCSEの検証・テストセットのミニバッチを作成"""
    # トークナイザを適用する
    tokenized_texts = tokenizer(
        [example["text"] for example in examples],
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )  # type: ignore

    # データセットに付与されたラベル配列のTensorを作成する
    label = torch.tensor([example["labels"] for example in examples])

    return {
        "tokenized_text": tokenized_texts,
        "labels": label,
    }


def unsup_train_collate_fn(
    examples: list[dict],
) -> dict[str, BatchEncoding | Tensor]:
    """マルチラベル訓練セットのミニバッチを作成"""
    # ミニバッチに含まれる文にトークナイザを適用する
    tokenized_texts = tokenizer(
        [example["text"] for example in examples],
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )  # type: ignore

    # 文と文の類似度行列における正例ペアの位置を示すTensorを作成する
    # 行列のi行目の事例（文）に対してi列目の事例（文）との組が正例ペアとなる
    labels = torch.arange(len(examples))

    return {
        "tokenized_texts_1": tokenized_texts,
        "tokenized_texts_2": tokenized_texts,
        "labels": labels,
    }


def sup_scl_train_collate_fn(
    examples: list[dict],
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
        max_length=512,
        return_tensors="pt",
    )  # type: ignore
    tokenized_texts2 = tokenizer(
        [example["same_label_text"] for example in examples],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )  # type: ignore

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
        max_length=512,
        return_tensors="pt",
    )  # type: ignore
    tokenized_texts2 = tokenizer(
        [example["same_label_text"] for example in examples],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )  # type: ignore

    labels = []
    for i in range(len(examples)):
        labels.append(torch.tensor(same_label_index[i]))

    return {
        "tokenized_texts_1": tokenized_texts1,
        "tokenized_texts_2": tokenized_texts2,
        "label_list": [example["labels"] for example in examples],
        "labels": labels,
    }
