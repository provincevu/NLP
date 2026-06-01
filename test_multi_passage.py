import json
import random
from collections import Counter

FILE_PATH = "data/splits/train.json"

BATCH_SIZE = 128
N_RUNS = 200


with open(FILE_PATH, "r", encoding="utf-8") as f:
    items = json.load(f)

all_batch_ratio = []
all_sample_ratio = []
all_avg_positive = []

for seed in range(N_RUNS):

    shuffled = items.copy()

    random.seed(seed)
    random.shuffle(shuffled)

    num_batches = 0
    batches_with_duplicate = 0

    total_samples = 0
    samples_with_multi_positive = 0

    total_positive_per_sample = 0

    for start in range(0, len(shuffled), BATCH_SIZE):

        batch = shuffled[start:start + BATCH_SIZE]

        pids = [x["positive_passage_id"] for x in batch]
        cnt = Counter(pids)

        if any(v > 1 for v in cnt.values()):
            batches_with_duplicate += 1

        for pid in pids:
            positives = cnt[pid]

            total_positive_per_sample += positives

            if positives > 1:
                samples_with_multi_positive += 1

        total_samples += len(batch)
        num_batches += 1

    all_batch_ratio.append(
        batches_with_duplicate / num_batches
    )

    all_sample_ratio.append(
        samples_with_multi_positive / total_samples
    )

    all_avg_positive.append(
        total_positive_per_sample / total_samples
    )

print("=" * 60)
print(f"Runs: {N_RUNS}")
print(f"Batch có multi-positive : {sum(all_batch_ratio)/N_RUNS:.2%}")
print(f"Sample có multi-positive: {sum(all_sample_ratio)/N_RUNS:.2%}")
print(f"Avg positives/sample    : {sum(all_avg_positive)/N_RUNS:.3f}")
print("=" * 60)