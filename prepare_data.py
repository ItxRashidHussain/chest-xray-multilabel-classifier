"""STEP 1 - Find the dataset, build multi-hot labels, make a patient-level split.

Run:  python prepare_data.py

It auto-detects the label CSV and the image files no matter how the Kaggle
download was unzipped, then writes artifacts/splits.csv.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

import config
from dataset import labels_to_vector


def find_csv():
    """Locate the labels CSV (sample dataset uses sample_labels.csv; full set uses Data_Entry_2017.csv)."""
    candidates = (list(config.DATA_DIR.rglob("sample_labels.csv"))
                  + list(config.DATA_DIR.rglob("Data_Entry_2017.csv")))
    if not candidates:
        sys.exit(f"ERROR: no label CSV found under {config.DATA_DIR}\n"
                 "Download the dataset first (see README.md).")
    return candidates[0]


def build_image_index():
    """Map every image filename -> full path, regardless of folder nesting."""
    index = {p.name: str(p) for p in config.DATA_DIR.rglob("*.png")}
    # a few mirrors of the dataset ship JPGs; include them too
    index.update({p.name: str(p) for p in config.DATA_DIR.rglob("*.jpg")})
    return index


def main():
    csv_path = find_csv()
    print(f"Labels CSV : {csv_path}")
    df = pd.read_csv(csv_path)

    img_index = build_image_index()
    print(f"Images found on disk : {len(img_index)}")

    df["path"] = df["Image Index"].map(img_index)
    missing = int(df["path"].isna().sum())
    if missing:
        print(f"WARNING: {missing} rows have no image file -> dropped")
    df = df.dropna(subset=["path"]).reset_index(drop=True)
    if len(df) == 0:
        sys.exit("ERROR: no CSV rows matched an image file. Check the data folder.")

    # Build the 14 binary label columns from the 'Finding Labels' text column.
    label_matrix = np.stack([labels_to_vector(s) for s in df["Finding Labels"]])
    for i, name in enumerate(config.DISEASE_LABELS):
        df[name] = label_matrix[:, i].astype(int)

    # ---- Patient-level split (no patient appears in two splits) ----
    if "Patient ID" in df.columns:
        groups = df["Patient ID"]
    else:
        groups = pd.Series(np.arange(len(df)), index=df.index)

    gss1 = GroupShuffleSplit(n_splits=1, test_size=config.TEST_FRACTION,
                             random_state=config.RANDOM_SEED)
    trainval_idx, test_idx = next(gss1.split(df, groups=groups))

    val_ratio = config.VAL_FRACTION / (1.0 - config.TEST_FRACTION)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_ratio,
                             random_state=config.RANDOM_SEED)
    tv = df.iloc[trainval_idx]
    rel_train, rel_val = next(gss2.split(tv, groups=groups.iloc[trainval_idx]))

    df["split"] = "train"
    df.loc[tv.iloc[rel_val].index, "split"] = "val"
    df.loc[df.iloc[test_idx].index, "split"] = "test"

    keep = ["Image Index", "path", "split"] + config.DISEASE_LABELS
    df[keep].to_csv(config.SPLITS_CSV, index=False)

    print("\nSplit sizes:")
    print(df["split"].value_counts().to_string())
    print("\nPositive samples per disease:")
    print(df[config.DISEASE_LABELS].sum().sort_values(ascending=False).to_string())
    print(f"\nSaved -> {config.SPLITS_CSV}")
    print("Next: python extract_features.py")


if __name__ == "__main__":
    main()
