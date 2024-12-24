import numpy as np
from xgb.utils.cls import execute_cls
from xgb.utils.embedding import embed_openai, embed_e5
from dataset.main import load_data
from transformers.trainer_utils import set_seed
import argparse

# パーサーを作成
parser = argparse.ArgumentParser(description="XGB model")
# 引数を定義
parser.add_argument(
    "--model_name", type=str, required=True, choices=["openai", "e5"], help="Model name"
)
parser.add_argument(
    "--dataset_name",
    type=str,
    required=True,
    choices=["wrime", "eurlex", "semeval-en", "semeval-ar", "semeval-es"],
    help="Task type",
)

# 引数を解析
args = parser.parse_args()

print("=" * 40)
print("Arguments")
print("=" * 40)
print(f"model_name: {args.model_name}")
print(f"dataset_name: {args.dataset_name}")

# 乱数のシードを設定する
set_seed(42)
# データ読み込み
train_dataset, valid_dataset, test_dataset = load_data(args.dataset_name)
# データの作成
if args.model_name == "e5":
    X_train = embed_e5(train_dataset["text"])  # type: ignore
    X_test = embed_e5(test_dataset["text"])  # type: ignore
elif args.model_name == "openai":
    X_train = embed_openai(train_dataset["text"])
    X_test = embed_openai(test_dataset["text"])
y_train = train_dataset["labels"]  # type: ignore
y_test = test_dataset["labels"]  # type: ignore
# `y_train` と `y_val` を NumPy 配列に変換
y_train = np.array(y_train)
y_test = np.array(y_test)

# モデルの学習
result = execute_cls(
    X_train,
    y_train,
    X_test,
    y_test,
    test_dataset["text"],
    model_name=args.model_name,
    type="base",
    dataset_name=args.dataset_name,
)
print(result)
