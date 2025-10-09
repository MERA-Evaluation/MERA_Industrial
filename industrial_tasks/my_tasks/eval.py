import json
import pandas as pd
from pathlib import Path

# 1) locate your samples file (adjust the glob if needed)
sample_path = next(Path("results").glob("*_samples.jsonl"))

# 2) load into a DataFrame
records = []
with open(sample_path, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        # assume metadata.subset holds your subset name
        records.append({
            "subset": row["metadata"]["subset"],
            "correct": row["prediction"].strip().lower() == row["target"].strip().lower(),
        })
df = pd.DataFrame(records)

# 3) compute accuracy per subset
metrics_by_subset = df.groupby("subset")["correct"].mean()
print(metrics_by_subset)
