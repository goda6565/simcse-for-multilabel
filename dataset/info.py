from collections import Counter
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import AutoTokenizer
from dataset.wrime import load_wrime
from dataset.semeval import load_semeval
from dataset.eurlex import load_eurlex
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")


def visualize_text_length(dataset: Dataset, dataset_name: str, name: str) -> None:
    """データセット中のテキストのトークン数の分布をグラフとして描画"""
    # データセット中のテキストの長さを数える
    length_counter = Counter()
    sum_length = 0
    for data in tqdm(dataset):
        length = len(tokenizer.tokenize(data["text"]))  # type: ignore
        sum_length += length
        length_counter[length] += 1
    print(f"平均系列長 {dataset_name} {name}: {sum_length / len(dataset):.2f}")
    # length_counterの値から棒グラフを描画する
    plt.title(f"{dataset_name} {name}")
    plt.bar(length_counter.keys(), length_counter.values())  # type: ignore
    plt.xlabel("num_token")
    plt.ylabel("num_data")
    plt.savefig(f"dataset/graph/{dataset_name}/{name}.png")
    plt.close()


if __name__ == "__main__":
    train_dataset, valid_dataset, test_dataset = load_wrime()
    visualize_text_length(train_dataset, "wrime", "train")
    visualize_text_length(valid_dataset, "wrime", "valid")
    visualize_text_length(test_dataset, "wrime", "test")
    train, valid, test = load_eurlex()
    visualize_text_length(train, "eurlex", "train")
    visualize_text_length(valid, "eurlex", "valid")
    visualize_text_length(test, "eurlex", "test")
    train, valid, test = load_semeval("en")
    visualize_text_length(train, "semeval-en", "train")
    visualize_text_length(valid, "semeval-en", "valid")
    visualize_text_length(test, "semeval-en", "test")
    train, valid, test = load_semeval("ar")
    visualize_text_length(train, "semeval-ar", "train")
    visualize_text_length(valid, "semeval-ar", "valid")
    visualize_text_length(test, "semeval-ar", "test")
    train, valid, test = load_semeval("es")
    visualize_text_length(train, "semeval-es", "train")
    visualize_text_length(valid, "semeval-es", "valid")
    visualize_text_length(test, "semeval-es", "test")
