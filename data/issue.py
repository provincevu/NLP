import json
from collections import Counter


TRAIN_PATH = "new/all_questions.json"


with open(TRAIN_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

cnt = Counter()

for item in data:
    pid = item["positive_passage_id"]
    cnt[pid] += 1

num_questions = len(data)
num_passages = len(cnt)

print("=" * 50)
print(f"Total questions           : {num_questions}")
print(f"Unique passages           : {num_passages}")
print(f"Avg questions / passage   : {num_questions / num_passages:.2f}")
print(f"Max questions / passage   : {max(cnt.values())}")
print(f"Min questions / passage   : {min(cnt.values())}")
print("=" * 50)

print("\nTop 60 passages có nhiều question nhất:\n")

for pid, n in cnt.most_common(60):
    print(f"{pid:<30} {n}")