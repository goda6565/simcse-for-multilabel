import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)


def embed_bert(
    texts: list[str],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    encoder: nn.Module,
    batch_size: int = 64,
) -> np.ndarray:
    """bert SimCSEのモデルを用いてテキストの埋め込みを計算"""
    embeddings = []

    # バッチごとにテキストを処理
    for i in tqdm(
        range(0, len(texts), batch_size), desc="Embedding texts", ncols=100
    ):  # tqdmで進捗表示
        batch_texts = texts[i : i + batch_size]

        # テキストにトークナイザを適用
        tokenized_texts = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to("cuda:0")

        # トークナイズされたテキストをベクトルに変換
        with torch.inference_mode():
            with torch.amp.autocast("cuda"):  # type: ignore
                encoded_texts = encoder(**tokenized_texts).last_hidden_state[:, 0]

        # ベクトルをNumPyのarrayに変換
        emb = encoded_texts.cpu().numpy().astype(np.float32)
        embeddings.append(emb)

    # バッチ処理した埋め込みを結合
    return np.concatenate(embeddings, axis=0)


def embed_l2v(
    texts: list[str],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    decoder: nn.Module,
    batch_size: int = 64,
) -> np.ndarray:
    """l2v SimCSEのモデルを用いてテキストの埋め込みを計算"""
    embeddings = []

    # バッチごとにテキストを処理
    for i in tqdm(
        range(0, len(texts), batch_size), desc="Embedding texts", ncols=100
    ):  # tqdmで進捗表示
        batch_texts = texts[i : i + batch_size]

        # テキストにトークナイザを適用
        tokenized_texts = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to("cuda:0")

        # トークナイズされたテキストをベクトルに変換
        with torch.inference_mode():
            with torch.amp.autocast("cuda"):  # type: ignore
                decoded_texts = decoder(**tokenized_texts)
                # 各文ごとにベクトルを作成し、スタックする
                encoded_texts = torch.stack(
                    [
                        decoded_texts.last_hidden_state[j, :, :].mean(dim=0)
                        for j in range(decoded_texts.last_hidden_state.size(0))
                    ]
                )

        # ベクトルをNumPyのarrayに変換
        emb = encoded_texts.cpu().numpy().astype(np.float32)
        embeddings.append(emb)

    # バッチ処理した埋め込みを結合
    return np.concatenate(embeddings, axis=0)
