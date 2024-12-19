from datasets import load_dataset, Dataset
from typing import Literal


def label_2_onehot(example):
    """ラベルのインデックスをワンホットベクトルに変換する"""
    num_class = 11
    label = {k: v for k, v in example.items() if k not in ["Tweet", "ID"]}
    onehot = [0] * num_class  # 全て0のリストを作成
    for i, target in enumerate(label.keys()):
        if label[target]:
            onehot[i] = 1
    return {"text": example["Tweet"], "labels": onehot}


# データセットを読み込む
def load_semeval(lang: Literal["en", "ar", "es"]) -> tuple[Dataset, Dataset, Dataset]:
    if lang == "en":
        dataset = load_dataset(
            "SemEvalWorkshop/sem_eval_2018_task_1",
            "subtask5.english",
            trust_remote_code=True,
        )
    elif lang == "ar":
        dataset = load_dataset(
            "SemEvalWorkshop/sem_eval_2018_task_1",
            "subtask5.arabic",
            trust_remote_code=True,
        )
    elif lang == "es":
        dataset = load_dataset(
            "SemEvalWorkshop/sem_eval_2018_task_1",
            "subtask5.spanish",
            trust_remote_code=True,
        )

    train_dataset = dataset["train"].map(  # type: ignore
        label_2_onehot,
        remove_columns=dataset["train"].column_names,  # type: ignore
    )
    valid_dataset = dataset["validation"].map(  # type: ignore
        label_2_onehot,
        remove_columns=dataset["train"].column_names,  # type: ignore
    )
    test_dataset = dataset["test"].map(  # type: ignore
        label_2_onehot,
        remove_columns=dataset["train"].column_names,  # type: ignore
    )
    return train_dataset, valid_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, valid_dataset, test_dataset = load_semeval("en")
    print(train_dataset[0])
    print(valid_dataset[0])
    print(test_dataset[0])
    train_dataset, valid_dataset, test_dataset = load_semeval("ar")
    print(train_dataset[0])
    print(valid_dataset[0])
    print(test_dataset[0])
    train_dataset, valid_dataset, test_dataset = load_semeval("es")
    print(train_dataset[0])
    print(valid_dataset[0])
    print(test_dataset[0])
    print("データの読み込みが完了しました。")
