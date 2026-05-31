import json
import random
from pathlib import Path
from collections import defaultdict


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def stratified_split_by_passage(
    in_path="data/new/all_questions.json",
    out_dir="data/splits",
    train_ratio=0.9,
    seed=42
):
    random.seed(seed)

    items = read_json(in_path)

    # group theo passage
    groups = defaultdict(list)

    for item in items:
        pid = item["positive_passage_id"]
        groups[pid].append(item)

    train_items = []
    val_items = []

    print(f"số passage: {len(groups)}")

    for pid, questions in groups.items():

        random.shuffle(questions)

        n = len(questions)

        # ít nhất 1 mẫu val nếu passage có >= 2 câu
        n_train = int(n * train_ratio)

        if n >= 2:
            n_train = min(n_train, n - 1)

        train_items.extend(questions[:n_train])
        val_items.extend(questions[n_train:])

    random.shuffle(train_items)
    random.shuffle(val_items)

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    write_json(train_items, f"{out_dir}/train.json")
    write_json(val_items, f"{out_dir}/val.json")

    print("=" * 50)
    print(f"Total : {len(items)}")
    print(f"Train : {len(train_items)}")
    print(f"Val   : {len(val_items)}")
    print("=" * 50)

    # thống kê nhanh
    train_passages = len(set(x["positive_passage_id"] for x in train_items))
    val_passages = len(set(x["positive_passage_id"] for x in val_items))

    print(f"Train passages : {train_passages}")
    print(f"Val passages   : {val_passages}")

if __name__ == "__main__":
    stratified_split_by_passage()