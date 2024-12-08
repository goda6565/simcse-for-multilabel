from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader

from transformers import Trainer
from transformers.trainer_utils import set_seed

from utils.data_edit import label_count, freq_labeling, create_same_label_datasets
from train_configs.bert.sscl import training_args
from utils.metrics import compute_metrics
from bert.utils.collates import eval_collate_fn, sup_not_scl_train_collate_fn
from bert.models.sscl import SimCSEModel
from bert.utils.bert import setup_bert


# 乱数のシードを設定する
set_seed(42)

# データ読み込み
train_dataset = load_dataset("Harutiin/eurlex-for-bert", split="train")
valid_dataset = load_dataset("Harutiin/eurlex-for-bert", split="validation")
test_dataset = load_dataset("Harutiin/eurlex-for-bert", split="test")

# peように訓練データを再構成(positive_ensured)
pe_train_dataset = create_same_label_datasets(train_dataset)

# 頻度によってリラベリング
label_count_list = label_count(valid_dataset)
valid_dataset = valid_dataset.map(
    lambda example: freq_labeling(example, label_count_list)
)


# BERTモデルとトークナイザを初期化する
model, tokenizer = setup_bert()
# 教師なしSimCSEのモデルを初期化する
sscl_model = SimCSEModel(model, mlp_only_train=True)

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
    model=sscl_model,
    args=training_args,
    data_collator=sup_not_scl_train_collate_fn,
    train_dataset=pe_train_dataset,  # type: ignore
    eval_dataset=valid_dataset,  # type: ignore
    compute_metrics=compute_metrics,
)

# パラメータを連続にする
for param in sscl_model.parameters():
    param.data = param.data.contiguous()

# 教師なしSimCSEの訓練を行う
trainer.train()

# エンコーダを保存
encoder_path = "outputs/bert/sscl/encoder"
sscl_model.encoder.save_pretrained(encoder_path)
tokenizer.save_pretrained(encoder_path)