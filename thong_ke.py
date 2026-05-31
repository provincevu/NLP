import json
from collections import Counter


TRAIN_PATH = "data/splits/val.json"


with open(TRAIN_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

cnt = Counter()

for item in data:
    pid = item["positive_passage_id"]
    cnt[pid] += 1

num_questions = len(data)
num_passages = len(cnt)

print("=" * 50)
print(f"Tổng số câu hỏi           : {num_questions}")
print(f"tổng số passage           : {num_passages}")
print(f"Avg questions / passage   : {num_questions / num_passages:.2f}")
print(f"Max questions / passage   : {max(cnt.values())}")
print(f"Min questions / passage   : {min(cnt.values())}")
print("=" * 50)

print("\nTop 150 passages có nhiều question nhất:\n")
count = 0
for pid, n in cnt.most_common(150):
    count += 1
    print(f"{count}.{pid:<30} {n}")

# ghi tất cả các passage_id ra file csv để tiện kiểm tra
with open("passage_ids.csv", "w", encoding="utf-8") as f:
    for pid in cnt.keys():
        f.write(pid + "\n")