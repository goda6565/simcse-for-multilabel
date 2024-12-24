import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from openai import OpenAI
from config import setting
from sentence_transformers import SentenceTransformer


def embed_bert(
    texts: list[str],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    encoder: nn.Module,
    max_length: int,
    batch_size: int = 256,
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
            max_length=max_length,
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
    max_length: int,
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
            max_length=max_length,
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


client = OpenAI(api_key=setting.open_api_key)


def embed_openai(texts: list[str], batch_size=512) -> np.ndarray:
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding Data in Batches"):
        batch_texts = texts[i : i + batch_size]
        
        response = client.embeddings.create(
            input=batch_texts, model="text-embedding-3-large"
        )
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)


def embed_e5(texts: list[str], batch_size=512) -> np.ndarray:
    model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding Data in Batches"):
        batch_texts = texts[i : i + batch_size]
        batch_embeddings = model.encode(batch_texts)
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)
