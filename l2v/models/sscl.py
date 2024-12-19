import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import ModelOutput
from transformers import BatchEncoding
from utils.coefficient import simpson_coefficient


class SimCSEModel(nn.Module):
    """sscl SimCSEのモデル"""

    def __init__(
        self,
        model: nn.Module,
        temperature: float = 0.05,
    ):
        """モデルの初期化"""
        super().__init__()

        # モデル名からエンコーダを初期化する
        self.decoder = model
        # 交差エントロピー損失の計算時に使用する温度
        self.temperature = temperature

    def encode_texts(self, tokenized_texts: BatchEncoding) -> Tensor:
        """エンコーダを用いて文をベクトルに変換"""
        # トークナイズされた文をエンコーダに入力する
        decoded_texts = self.decoder(**tokenized_texts)

        # 各文ごとにベクトルを作成し、スタックする
        encoded_texts = torch.stack(
            [
                decoded_texts.last_hidden_state[j, :, :].mean(dim=0)
                for j in range(decoded_texts.last_hidden_state.size(0))
            ]
        )

        return encoded_texts

    def forward(self, **inputs) -> ModelOutput:
        """モデルの前向き計算（訓練と検証の両方に対応）"""

        # 訓練と検証の両方で使われる入力データを処理
        if "tokenized_text" in inputs:
            # 検証用の処理
            tokenized_text = inputs["tokenized_text"]
            labels = inputs["labels"]
            encoded_text = self.encode_texts(tokenized_text)
            return ModelOutput(loss=torch.tensor(0.0).to("cuda"), scores=encoded_text)

        # 訓練用の処理
        tokenized_texts_1 = inputs["tokenized_texts_1"]
        tokenized_texts_2 = inputs["tokenized_texts_2"]
        label = inputs["label_list"]
        labels = inputs["labels"]

        # 文ペアをベクトルに変換する
        encoded_texts_1 = self.encode_texts(tokenized_texts_1)
        encoded_texts_2 = self.encode_texts(tokenized_texts_2)

        # loss計算
        loss = 0
        for i in range(len(encoded_texts_1)):
            loss_i = 0
            # 分母の計算
            denominator = sum(
                torch.exp(
                    F.cosine_similarity(
                        encoded_texts_1[i].unsqueeze(0), encoded_texts_2[j].unsqueeze(0)
                    )
                    / self.temperature
                )
                for j in range(len(encoded_texts_1))
            )
            # 交差エントロピー損失を計算
            for s in labels[i]:
                # 同じバッチ内で他のサンプルとの比較
                simpson = simpson_coefficient(label[i], label[s])
                # コサイン類似度
                sim_ij = (
                    F.cosine_similarity(
                        encoded_texts_1[i].unsqueeze(0), encoded_texts_2[s].unsqueeze(0)
                    )
                    / self.temperature
                )
                sim_ij = torch.exp(sim_ij)
                # ロスの計算
                loss_i += simpson * torch.log(sim_ij / denominator)

            loss += -1 * loss_i / len(labels[i])

        return ModelOutput(loss=loss[0])  # type: ignore
