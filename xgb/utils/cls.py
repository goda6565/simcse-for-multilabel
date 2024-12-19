import wandb
import numpy as np
import xgboost as xgb
from typing import Literal
from xgb.utils.models import XGBScore
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

test_dataset = load_dataset("Harutiin/eurlex-for-bert", split="test")


def execute_cls(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: Literal["bert", "l2v", "openai"],
    type: Literal["unsup", "scl", "sscl", "jscl", "dscl", "openai"],
) -> XGBScore:
    # wandb run を開始
    run = wandb.init(project="xgb-cls", name=f"{model_name}-{type}")
    """XGBoostを使って分類を行う"""
    # モデルを初期化
    xgb.set_config(verbosity=1)
    model = xgb.XGBClassifier(tree_method="hist", device="cuda")
    # モデルを訓練
    model.fit(X_train, y_train)
    # 予測を行う
    pred = model.predict(X_test)

    # 評価指標を計算
    accuracy = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro", zero_division=0)
    macro_precision = precision_score(y_test, pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_test, pred, average="macro", zero_division=0)

    # 評価指標をログに記録
    wandb.log(
        {
            "accuracy": float(f"{accuracy:.5f}"),
            "macro_f1": float(f"{macro_f1:.5f}"),
            "macro_precision": float(f"{macro_precision:.5f}"),
            "macro_recall": float(f"{macro_recall:.5f}"),
        }
    )

    # wandb run を終了
    run.finish()

    # データセットを作成
    output_dataset = Dataset.from_dict(
        {
            "pred": pred,
            "y_test": y_test,
            "X_test": test_dataset["text"],  # type: ignore
        }
    )

    # CSV に保存
    output_dataset.to_csv(f"outputs/xgb/{model_name}/{type}.csv", index=False)

    return XGBScore(
        accuracy=float(f"{accuracy:.5f}"),
        macro_f1=float(f"{macro_f1:.5f}"),
        macro_precision=float(f"{macro_precision:.5f}"),
        macro_recall=float(f"{macro_recall:.5f}"),
    )
