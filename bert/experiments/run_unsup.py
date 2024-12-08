from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader

from transformers import Trainer
from transformers.trainer_utils import set_seed

from utils.data_edit import label_count, freq_labeling
from train_configs.bert.unsup import training_args
from utils.metrics import compute_metrics
from bert.utils.collates import eval_collate_fn, unsup_train_collate_fn
from bert.models.unsup import SimCSEModel
from bert.utils.bert import setup_bert

# 乱数のシードを設定する
set_seed(42)

# データ読み込み
train_dataset = load_dataset("Harutiin/eurlex-for-bert", split="train")
valid_dataset = load_dataset("Harutiin/eurlex-for-bert", split="validation")
test_dataset = load_dataset("Harutiin/eurlex-for-bert", split="test")

label_count_list = label_count(valid_dataset)
valid_dataset = valid_dataset.map(
    lambda example: freq_labeling(example, label_count_list)
)

# BERTモデルとトークナイザを初期化する
model, tokenizer = setup_bert()
# 教師なしSimCSEのモデルを初期化する
unsup_model = SimCSEModel(model, mlp_only_train=True)


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
            collate_fn=eval_collate_fn,
            pin_memory=True,
        )


# Trainerを初期化する
trainer = SimCSETrainer(
    model=unsup_model,
    args=training_args,
    data_collator=unsup_train_collate_fn,
    train_dataset=train_dataset,  # type: ignore
    eval_dataset=valid_dataset,  # type: ignore
    compute_metrics=compute_metrics,
)

# パラメータを連続にする
for param in unsup_model.parameters():
    param.data = param.data.contiguous()
print(type(unsup_model).__name__)

# 教師なしSimCSEの訓練を行う
trainer.train()

# エンコーダを保存
encoder_path = "outputs/bert/unsup/encoder"
unsup_model.encoder.save_pretrained(encoder_path)
tokenizer.save_pretrained(encoder_path)
