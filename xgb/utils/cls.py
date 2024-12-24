import os
import wandb
import pickle
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
    test_text: list[str],
    model_name: Literal["bert", "l2v", "openai", "e5"],
    type: Literal["unsup", "scl", "sscl", "jscl", "dscl", "base"],
    dataset_name: Literal["wrime", "eurlex", "semeval-en", "semeval-ar", "semeval-es"],
) -> XGBScore:
    # wandb run を開始
    run = wandb.init(project=f"xgb-{dataset_name}", name=f"{model_name}-{type}")
    """XGBoostを使って分類を行う"""
    # モデルを初期化
    xgb.set_config(verbosity=1)
    print(f"dimention: {X_train.shape[1]}")
    model = xgb.XGBClassifier(tree_method="hist", device="cuda")
    # モデルを訓練
    model.fit(X_train, y_train)
    # 予測を行う
    pred = model.predict(X_test)

    # 評価指標を計算
    accuracy = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_test, pred, average="micro", zero_division=0)
    macro_precision = precision_score(y_test, pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_test, pred, average="macro", zero_division=0)
    # pickle ファイルにデータを保存
    output_path = f"outputs/{model_name}/{type}/{dataset_name}"
    os.makedirs(output_path, exist_ok=True)
    with open(f"outputs/{model_name}/{type}/{dataset_name}/xgb.pkl", "wb") as f:
        pickle.dump(test_text, f)
        pickle.dump(y_test, f)
        pickle.dump(pred, f)

    # 評価指標をログに記録
    wandb.log(
        {
            "accuracy": float(f"{accuracy:.5f}"),
            "macro_f1": float(f"{macro_f1:.5f}"),
            "micro_f1": float(f"{micro_f1:.5f}"),
            "macro_precision": float(f"{macro_precision:.5f}"),
            "macro_recall": float(f"{macro_recall:.5f}"),
        }
    )

    # wandb run を終了
    run.finish()

    return XGBScore(
        accuracy=float(f"{accuracy:.5f}"),
        macro_f1=float(f"{macro_f1:.5f}"),
        micro_f1=float(f"{micro_f1:.5f}"),
        macro_precision=float(f"{macro_precision:.5f}"),
        macro_recall=float(f"{macro_recall:.5f}"),
    )
