import wandb
import numpy as np
import xgboost as xgb
from typing import Literal
from xgb.utils.models import XGBScore
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def execute_cls(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: Literal["bert", "l2v"],
    type: Literal["unsup", "scl", "sscl", "jscl", "dscl"],
) -> XGBScore:
    # wandb run を開始
    run = wandb.init(project="xgb-cls", name=f"{model_name}-{type}")
    """XGBoostを使って分類を行う"""
    # モデルを初期化
    model = xgb.XGBClassifier(tree_method="hist", device="cuda", verbosity=1)
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

    return XGBScore(
        accuracy=float(f"{accuracy:.5f}"),
        macro_f1=float(f"{macro_f1:.5f}"),
        macro_precision=float(f"{macro_precision:.5f}"),
        macro_recall=float(f"{macro_recall:.5f}"),
    )
