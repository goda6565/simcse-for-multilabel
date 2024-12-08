import numpy as np
from xgb.utils.cls import execute_cls
from xgb.utils.load_model import load_l2v
from xgb.utils.embedding import embed_l2v
from datasets import load_dataset
from transformers.trainer_utils import set_seed

# モデル読み込み
model_path = "outputs/l2v/scl/decoder"
decoder, tokenizer = load_l2v(model_path)

# 読み込んだモデルをGPUに
device = "cuda:0"
decoder = decoder.to(device)

# 乱数のシードを設定する
set_seed(42)
# データ読み込み
train_dataset = load_dataset("Harutiin/eurlex-for-bert", split="train")
test_dataset = load_dataset("Harutiin/eurlex-for-bert", split="test")


# データの作成
# 訓練
X_train = embed_l2v(train_dataset["text"], tokenizer, decoder) # type: ignore
y_train = train_dataset["labels"] # type: ignore
# テスト
X_test = embed_l2v(test_dataset["text"], tokenizer, decoder) # type: ignore
y_test = test_dataset["labels"] # type: ignore
# `y_train` と `y_val` を NumPy 配列に変換
y_train = np.array(y_train)
y_test = np.array(y_test)

# モデルの学習
result = execute_cls(X_train, y_train, X_test, y_test, model_name="l2v", type="scl")
print(result)