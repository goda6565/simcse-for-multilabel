from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import Trainer
from transformers.trainer_utils import set_seed
from utils.data_edit import label_count, freq_labeling, create_same_label_datasets
from utils.metrics import compute_metrics
from l2v.utils.collates import eval_collate_fn, sup_scl_train_collate_fn
from train_configs.l2v.scl import training_args
from l2v.models.scl import SimCSEModel
from l2v.utils.l2v import setup_l2v
import wandb

wandb.init(project="simcse-l2v", name="unscl")


# 乱数のシードを設定する
set_seed(42)

# データ読み込み
train_dataset = load_dataset("Harutiin/eurlex-for-bert", split="train")
valid_dataset = load_dataset("Harutiin/eurlex-for-bert", split="validation")
test_dataset = load_dataset("Harutiin/eurlex-for-bert", split="test")

# peように訓練データを再構成(positive_ensured)
train_dataset = create_same_label_datasets(train_dataset)

# 頻度によってリラベリング
label_count_list = label_count(valid_dataset)
valid_dataset = valid_dataset.map(
    lambda example: freq_labeling(example, label_count_list)
)

model, tokenizer = setup_l2v()
# 教師なしSimCSEのモデルを初期化する
scl_model = SimCSEModel(model=model)


# Trainerを初期化する
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
            collate_fn=eval_collate_fn,  # type: ignore
            pin_memory=True,
        )


# Trainerを初期化する
trainer = SimCSETrainer(
    model=scl_model,
    args=training_args,
    data_collator=sup_scl_train_collate_fn,  # type: ignore
    train_dataset=train_dataset,  # type: ignore
    eval_dataset=valid_dataset,  # type: ignore
    compute_metrics=compute_metrics,
)

# パラメータを連続にする
for param in scl_model.parameters():
    param.data = param.data.contiguous()
print(type(scl_model).__name__)

# 教師なしSimCSEの訓練を行う
trainer.train()

# デコーダを保存
decoder_path = "outputs/l2v/scl/decoder"
scl_model.decoder.save_pretrained(decoder_path)
tokenizer.save_pretrained(decoder_path)  # type: ignore
