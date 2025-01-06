import numpy as np
from xgb.utils.cls import execute_cls
from xgb.utils.embedding import embed_bert, embed_l2v
from xgb.utils.load_model import load_bert, load_l2v
from dataset.main import load_data
from transformers.trainer_utils import set_seed
import argparse

# パーサーを作成
parser = argparse.ArgumentParser(description="XGB model")
# 引数を定義
parser.add_argument(
    "--model_name", type=str, required=True, choices=["l2v", "bert"], help="Model name"
)
parser.add_argument(
    "--model_type",
    type=str,
    required=True,
    choices=["scl", "dscl", "jscl", "sscl", "base"],
    help="Model type",
)
parser.add_argument(
    "--dataset_name",
    type=str,
    required=True,
    choices=["wrime", "eurlex", "semeval-en", "semeval-ar", "semeval-es"],
    help="Task type",
)
parser.add_argument(
    "--max_length", type=int, required=True, help="Maximum length of input text"
)
parser.add_argument("--output_dir", type=str, required=True, help="Output directory")

# 引数を解析
args = parser.parse_args()

print("=" * 40)
print("Arguments")
print("=" * 40)
print(f"model_name: {args.model_name}")
print(f"model_type: {args.model_type}")
print(f"dataset_name: {args.dataset_name}")
print(f"max_length: {args.max_length}")

# モデル読み込み
model_path = args.output_dir
if args.model_name == "l2v" and args.model_type == "base":
    decoder, tokenizer = load_l2v(f"{model_path}/decoder", True)
    model = decoder
elif args.model_name == "l2v":
    decoder, tokenizer = load_l2v(f"{model_path}/decoder")
    model = decoder
elif args.model_type == "base":
    encoder, tokenizer = load_bert(f"{model_path}/encoder", True)
    model = encoder
else:
    encoder, tokenizer = load_bert(f"{model_path}/encoder")
    model = encoder

# 読み込んだモデルをGPUに
device = "cuda:0"
model = model.to(device)

# 乱数のシードを設定する
set_seed(42)
# データ読み込み
train_dataset, valid_dataset, test_dataset = load_data(args.dataset_name)


# データの作成
if args.model_name == "l2v":
    # 訓練
    X_train = embed_l2v(train_dataset["text"], tokenizer, decoder, args.max_length)  # type: ignore
    y_train = train_dataset["labels"]  # type: ignore
    # テスト
    X_test = embed_l2v(test_dataset["text"], tokenizer, decoder, args.max_length)  # type: ignore
    y_test = test_dataset["labels"]  # type: ignore
else:
    # 訓練
    X_train = embed_bert(train_dataset["text"], tokenizer, encoder, args.max_length)  # type: ignore
    y_train = train_dataset["labels"]  # type: ignore
    # テスト
    X_test = embed_bert(test_dataset["text"], tokenizer, encoder, args.max_length)  # type: ignore
    y_test = test_dataset["labels"]  # type: ignore

# `y_train` と `y_val` を NumPy 配列に変換
y_train = np.array(y_train)
y_test = np.array(y_test)

path = args.output_dir
batch_size = path.split("/")[-1]

# モデルの学習
result = execute_cls(
    X_train,
    y_train,
    X_test,
    y_test,
    test_dataset["text"],
    model_name=args.model_name,
    type=args.model_type,
    dataset_name=args.dataset_name,
    batch_size=batch_size,
)
print(result)
