import json
import random
from pathlib import Path

def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(items, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

def split_questions(
    in_path="data/questions.json",
    out_dir="data/splits",
    train_ratio=0.9,
    seed=42
):

    items = read_json(in_path)

    random.seed(seed)
    random.shuffle(items)

    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * (1 - train_ratio))  # các phần còn lại sẽ là val

    train_items = items[:n_train]
    val_items = items[n_train:n_train + n_val]
    test_items = items[n_train + n_val:]

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    write_json(train_items, f"{out_dir}/train.json")
    write_json(val_items, f"{out_dir}/val.json")
    write_json(test_items, f"{out_dir}/test.json")

    print(f"Total: {n}")
    print(f"Train: {len(train_items)}")
    print(f"Val:   {len(val_items)}")
    print(f"Test:  {len(test_items)}")
    print(f"Saved to: {out_dir}")

if __name__ == "__main__":
    split_questions()