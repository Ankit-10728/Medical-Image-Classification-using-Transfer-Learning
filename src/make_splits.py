
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

DATA_DIR = Path("data/raw")
OUT_PATH = Path("data/splits.csv")
SEED = 42  # fixed seed = same split every time you run this

df = pd.read_csv(DATA_DIR / "HAM10000_metadata.csv")


sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)
trainval_idx, test_idx = next(sgkf.split(df, df["dx"], groups=df["lesion_id"]))
df["split"] = "train"
df.loc[test_idx, "split"] = "test"

# Step 2: from the remaining data, hold out another ~14% as VALIDATION (1 of 6 folds)
trainval = df.iloc[trainval_idx]
sgkf = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=SEED)
_, val_sub = next(sgkf.split(trainval, trainval["dx"], groups=trainval["lesion_id"]))
df.loc[trainval.index[val_sub], "split"] = "val"

# Step 3: safety check - no lesion may appear in more than one split
leaks = df.groupby("lesion_id")["split"].nunique().gt(1).sum()
assert leaks == 0, f"{leaks} lesions appear in more than one split!"

print("Images per split:")
print(df["split"].value_counts(), "\n")
print("Class share inside each split (columns should look similar):")
print(pd.crosstab(df["dx"], df["split"], normalize="columns").round(3))

df[["image_id", "lesion_id", "dx", "split"]].to_csv(OUT_PATH, index=False)
print(f"\nSaved {OUT_PATH}")