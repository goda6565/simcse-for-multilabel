from typing import Literal
from dataset.wrime import load_wrime
from dataset.eurlex import load_eurlex
from dataset.semeval import load_semeval


def load_data(
    dataset_name: Literal["wrime", "eurlex", "semeval-en", "semeval-ar", "semeval-es"],
):
    if dataset_name == "wrime":
        train, valid, test = load_wrime()
    elif dataset_name == "eurlex":
        train, valid, test = load_eurlex()
    elif dataset_name == "semeval-en":
        train, valid, test = load_semeval("en")
    elif dataset_name == "semeval-ar":
        train, valid, test = load_semeval("ar")
    elif dataset_name == "semeval-es":
        train, valid, test = load_semeval("es")
    return train, valid, test


if __name__ == "__main__":
    train, valid, test = load_data("wrime")
    print(train[0])
    print(valid[0])
    print(test[0])
    print("データの読み込みが完了しました。")
    train, valid, test = load_data("eurlex")
    print(train[0])
    print(valid[0])
    print(test[0])
    print("データの読み込みが完了しました。")
    train, valid, test = load_data("semeval-en")
    print(train[0])
    print(valid[0])
    print(test[0])
    train, valid, test = load_data("semeval-ar")
    print(train[0])
    print(valid[0])
    print(test[0])
    train, valid, test = load_data("semeval-es")
    print(train[0])
    print(valid[0])
    print(test[0])
    print("データの読み込みが完了しました。")
