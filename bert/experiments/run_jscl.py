from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import set_seed
from dataset.main import load_data
from utils.data_edit import create_same_label_datasets
from utils.metrics import compute_metrics
from bert.utils.collates import eval_collate_fn, sup_not_scl_train_collate_fn
from bert.models.jscl import SimCSEModel
from bert.utils.bert import setup_bert
import argparse
import wandb


# 乱数のシードを設定する
set_seed(42)

# パーサーを作成
parser = argparse.ArgumentParser(description="Train BERT SimCSE model")
# 引数を定義
parser.add_argument(
    "--output_dir", type=str, required=True, help="Path to save model output"
)
parser.add_argument(
    "--dataset_name",
    type=str,
    required=True,
    choices=["wrime", "eurlex", "semeval-en", "semeval-ar", "semeval-es"],
    help="Task type",
)
parser.add_argument(
    "--per_device_batch_size", type=int, default=64, help="Batch size per device"
)
parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
parser.add_argument(
    "--max_length", type=int, default=512, help="Maximum length of input text"
)

# 引数を解析
args = parser.parse_args()

wandb.init(project=args.dataset_name, name=f"bert-jscl batch size: {args.per_device_batch_size}")

print("=" * 40)
print("Arguments")
print("=" * 40)
print(f"output_dir: {args.output_dir}")
print(f"dataset_name: {args.dataset_name}")
print(f"per_device_batch_size: {args.per_device_batch_size}")
print(f"learning_rate: {args.learning_rate}")
print(f"max_length: {args.max_length}")

# データ読み込み
train_dataset, valid_dataset, test_dataset = load_data(args.dataset_name)

# peように訓練データを再構成(positive_ensured)
pe_train_dataset = create_same_label_datasets(train_dataset)

step_size = int(len(pe_train_dataset) / args.per_device_batch_size / 5)

print(f"record_steps: {step_size}")
print("=" * 40)


# BERTモデルとトークナイザを初期化する
model, tokenizer = setup_bert()
# 教師なしSimCSEのモデルを初期化する
jscl_model = SimCSEModel(model, mlp_only_train=True)


# 訓練設定
class SimCSETrainer(Trainer):
    """SimCSEの訓練に使用するTrainer"""

    def get_eval_dataloader(self, eval_dataset: Dataset | None = None) -> DataLoader:
        """
        検証・テストセットのDataLoaderでeval_collate_fnを使うように
        Trainerのget_eval_dataloaderをオーバーライド
        """
        if eval_dataset is None:
            eval_dataset = self.eval_dataset  # type: ignore

        return DataLoader(
            eval_dataset,  # type: ignore
            batch_size=64,
            collate_fn=lambda examples: eval_collate_fn(
                examples, max_length=args.max_length
            ),
            pin_memory=True,
        )


# 訓練のハイパーパラメータを設定する
training_args = TrainingArguments(
    output_dir=args.output_dir,  # 結果の保存先フォルダ
    per_device_train_batch_size=args.per_device_batch_size,  # 訓練時のバッチサイズ
    per_device_eval_batch_size=args.per_device_batch_size,  # 評価時のバッチサイズ
    learning_rate=args.learning_rate,  # 学習率
    num_train_epochs=1,  # 訓練エポック数（step数）
    evaluation_strategy="steps",  # 検証セットによる評価のタイミング
    eval_steps=step_size,  # 検証セットによる評価を行う訓練ステップ数の間隔
    logging_steps=step_size,  # ロギングを行う訓練ステップ数の間隔
    save_steps=step_size,  # チェックポイントを保存する訓練ステップ数の間隔
    save_total_limit=1,  # 保存するチェックポイントの最大数
    fp16=True,  # 自動混合精度演算の有効化
    load_best_model_at_end=True,  # 最良のモデルを訓練終了後に読み込むか
    metric_for_best_model="f1",  # 最良のモデルを決定する評価指標
    label_names=["labels"],  # ラベルを指定マルチラベル
    remove_unused_columns=False,  # データセットの不要フィールドを削除するか
    report_to="wandb",
)

# Trainerを初期化する
trainer = SimCSETrainer(
    model=jscl_model,
    args=training_args,
    data_collator=lambda examples: sup_not_scl_train_collate_fn(
        examples, max_length=args.max_length
    ),
    train_dataset=pe_train_dataset,  # type: ignore
    eval_dataset=valid_dataset,  # type: ignore
    compute_metrics=compute_metrics,
)

# パラメータを連続にする
for param in jscl_model.parameters():
    param.data = param.data.contiguous()

# 教師なしSimCSEの訓練を行う
trainer.train()

# エンコーダを保存
encoder_path = f"{args.output_dir}/encoder"
jscl_model.encoder.save_pretrained(encoder_path)
tokenizer.save_pretrained(encoder_path)
