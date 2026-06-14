# 🫁 Chest X-ray Multi-Label Disease Classifier

A computer-vision project that classifies **chest X-rays** into **14 possible
thoracic diseases** at once (multi-label). A single X-ray can show several
findings simultaneously (e.g. *Cardiomegaly + Effusion*), so the model predicts
an independent probability for each condition.

Built to run on a **CPU-only machine** using *transfer learning*: a frozen
pretrained CNN turns each image into a feature vector, and a small classifier
head is trained on those vectors.

---

## How it works (the CPU-friendly trick)

```
X-ray image ──► [ Frozen pretrained CNN ] ──► feature vector ──► [ Small head ] ──► 14 probabilities
                 (MobileNetV2, not trained)    (cached to disk)    (this is what we train)
```

Training a deep CNN from scratch on a CPU is impractical. Instead we:
1. Use a CNN already trained on millions of images as a fixed **feature extractor**.
2. Run every X-ray through it **once** and **cache** the resulting vectors.
3. Train only a tiny **classifier head** on those cached vectors — fast on CPU.

---

## Project structure

| File | Role |
|------|------|
| `config.py` | All settings: paths, the 14 labels, backbone, hyper-parameters |
| `model.py` | The frozen backbone + the trainable classifier head |
| `dataset.py` | Image preprocessing + multi-label parsing |
| `prepare_data.py` | **Step 1** — find data, build labels, patient-level train/val/test split |
| `extract_features.py` | **Step 2** — cache feature vectors (the one slow step) |
| `train.py` | **Step 3** — train the head, report ROC-AUC / F1, save the model |
| `app.py` | **Step 4** — Streamlit demo: upload an X-ray → predictions |
| `requirements.txt` | Dependencies |
| `data/` | Put the dataset here (git-ignored) |
| `artifacts/` | Generated splits, features, model, metrics (git-ignored) |

---

## Setup

```bash
pip install -r requirements.txt
```

### Get the dataset (NIH Chest X-ray *sample*, ≈5,606 images, ~5 GB)

First, one-time Kaggle setup (needed for **both** options below):
1. Make a free account at [kaggle.com](https://www.kaggle.com).
2. Kaggle → your avatar → **Settings → API → "Create New Token"** → downloads `kaggle.json`.
3. Move `kaggle.json` to `C:\Users\<you>\.kaggle\kaggle.json`
   (or set env vars `KAGGLE_USERNAME` and `KAGGLE_KEY`).

**Option A — API (recommended): resumable, no manual unzip**
```bash
python download_data.py
```
It downloads to kagglehub's cache and records the path so the pipeline finds it
automatically.

**Option B — manual zip**
Download the 5 GB zip from
<https://www.kaggle.com/datasets/nih-chest-xrays/sample> and unzip it into the
`data/` folder. The scripts search recursively, so the layout doesn't matter —
they just need to find `sample_labels.csv` and the `.png` images under `data/`.

---

## Run it (in order)

```bash
python download_data.py     # Step 0 (optional, recommended) — Kaggle API download
python prepare_data.py      # builds artifacts/splits.csv
python extract_features.py  # caches features (downloads backbone weights once; slowest step)
python train.py             # trains + prints test metrics, saves the model
streamlit run app.py        # opens the web demo at http://localhost:8501
```

---

## What to expect

- **Feature extraction** is the only slow part on CPU (one forward pass per
  image) and runs **once**. After that, training is quick.
- Reported metric is **per-class ROC-AUC** (the standard for this dataset)
  plus **macro-F1**. Expect ~0.70–0.80 AUC on the common classes — solid for a
  student project on the small sample set.
- Want higher accuracy and have time? In `config.py` set
  `BACKBONE = "densenet121"` (slower but stronger) and re-run from step 2.

---

## Notes for the report

- **Multi-label vs multi-class:** each disease is an independent yes/no
  (`sigmoid` + `BCEWithLogitsLoss`), not a single softmax choice.
- **Class imbalance:** rare diseases are up-weighted in the loss (`pos_weight`).
- **No data leakage:** the split is done **by patient**, so the same patient
  never appears in both training and test.

> ⚠️ Educational project only — **not** a medical device. Do not use for real diagnosis.
