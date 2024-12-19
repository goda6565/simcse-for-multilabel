from datasets import load_dataset, Dataset


def label_2_onehot(example):
    """ラベルのインデックスをワンホットベクトルに変換する"""
    num_class = 100
    label = example["labels"]
    onehot = [0] * num_class  # 全て0のリストを作成
    for idx in label:
        onehot[idx] = 1  # 該当するインデックスを1に設定
    example["labels"] = onehot  # onehotエンコーディングを追加
    return example


def load_eurlex() -> tuple[Dataset, Dataset, Dataset]:
    """eurlexを読み込む"""
    dataset = load_dataset("coastalcph/lex_glue", "eurlex", trust_remote_code=True)
    train_dataset = dataset["train"].map(label_2_onehot)  # type: ignore
    valid_dataset = dataset["validation"].map(label_2_onehot)  # type: ignore
    test_dataset = dataset["test"].map(label_2_onehot)  # type: ignore
    return train_dataset, valid_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, valid_dataset, test_dataset = load_eurlex()
    print(train_dataset[0])
    print(valid_dataset[0])
    print(test_dataset[0])
    print("データの読み込みが完了しました。")
