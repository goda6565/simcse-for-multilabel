from datasets import load_dataset, Dataset


def label_2_onehot(example):
    """ラベルのインデックスをワンホットベクトルに変換する"""
    num_class = 8
    label = example["avg_readers"]
    onehot = [0] * num_class  # 全て0のリストを作成
    for i, target in enumerate(label.keys()):
        if label[target] != 0:
            onehot[i] = 1
    return {"text": example["sentence"], "labels": onehot}


# データセットを読み込む
def load_wrime() -> tuple[Dataset, Dataset, Dataset]:
    dataset = load_dataset("shunk031/wrime", "ver1", trust_remote_code=True)
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
    train_dataset, valid_dataset, test_dataset = load_wrime()
    print(train_dataset[0])
    print(valid_dataset[0])
    print(test_dataset[0])
    print("データの読み込みが完了しました。")
