"""STEP 3 - Train the multi-label classifier head and evaluate it.

Run:  python train.py

Trains on the cached features (fast on CPU), keeps the best epoch by validation
macro-AUC, then reports per-class ROC-AUC + macro-F1 on the held-out test set
and saves the model for the demo app.
"""
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, f1_score

import config
from model import ClassifierHead


def load_split(split):
    X = np.load(config.FEATURES_DIR / f"X_{split}.npy")
    y = np.load(config.FEATURES_DIR / f"y_{split}.npy")
    return torch.from_numpy(X).float(), torch.from_numpy(y).float()


def evaluate(head, X, y):
    """Return (probabilities, {class: AUC}, macro_AUC)."""
    head.eval()
    with torch.no_grad():
        probs = torch.sigmoid(head(X)).numpy()
    y = y.numpy()
    aucs = {}
    for i, name in enumerate(config.DISEASE_LABELS):
        if len(np.unique(y[:, i])) == 2:  # AUC needs both a positive and a negative
            aucs[name] = roc_auc_score(y[:, i], probs[:, i])
    macro_auc = float(np.mean(list(aucs.values()))) if aucs else float("nan")
    return probs, aucs, macro_auc


def main():
    if not (config.FEATURES_DIR / "X_train.npy").exists():
        sys.exit("Run extract_features.py first.")

    torch.manual_seed(config.RANDOM_SEED)
    Xtr, ytr = load_split("train")
    Xval, yval = load_split("val")
    Xte, yte = load_split("test")
    feat_dim = Xtr.shape[1]
    print(f"feat_dim={feat_dim}  train={len(Xtr)}  val={len(Xval)}  test={len(Xte)}")

    head = ClassifierHead(feat_dim)

    # Class imbalance: rare diseases get a larger positive weight in the loss.
    pos = ytr.sum(dim=0)
    neg = len(ytr) - pos
    pos_weight = torch.clamp(neg / torch.clamp(pos, min=1.0), max=50.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(head.parameters(), lr=config.LEARNING_RATE,
                                 weight_decay=config.WEIGHT_DECAY)

    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=config.TRAIN_BATCH_SIZE,
                        shuffle=True, drop_last=True)  # drop_last avoids a size-1 BatchNorm batch

    best_auc, best_state = -1.0, None
    for epoch in range(1, config.EPOCHS + 1):
        head.train()
        running = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(head(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(xb)
        _, _, val_auc = evaluate(head, Xval, yval)
        print(f"epoch {epoch:02d}  loss {running/len(Xtr):.4f}  val_macroAUC {val_auc:.4f}")
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.clone() for k, v in head.state_dict().items()}

    # ---- Final test evaluation with the best epoch ----
    head.load_state_dict(best_state)
    probs, aucs, macro_auc = evaluate(head, Xte, yte)
    preds = (probs >= config.DECISION_THRESHOLD).astype(int)
    macro_f1 = f1_score(yte.numpy(), preds, average="macro", zero_division=0)

    lines = [
        "=== TEST RESULTS ===",
        f"Backbone               : {config.BACKBONE}",
        f"Best val macro-AUC     : {best_auc:.4f}",
        f"Test macro-AUC         : {macro_auc:.4f}",
        f"Test macro-F1 (@{config.DECISION_THRESHOLD}) : {macro_f1:.4f}",
        "",
        "Per-class ROC-AUC (high = better):",
    ]
    for name, v in sorted(aucs.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {name:20s} {v:.3f}")
    report = "\n".join(lines)
    print("\n" + report)
    config.METRICS_PATH.write_text(report)

    torch.save({
        "state_dict": head.state_dict(),
        "feat_dim": feat_dim,
        "labels": config.DISEASE_LABELS,
        "backbone": config.BACKBONE,
        "threshold": config.DECISION_THRESHOLD,
    }, config.MODEL_PATH)
    print(f"\nSaved model   -> {config.MODEL_PATH}")
    print(f"Saved metrics -> {config.METRICS_PATH}")
    print("Next: streamlit run app.py")


if __name__ == "__main__":
    main()
