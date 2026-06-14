"""STEP 3 - Train the multi-label classifier head and evaluate it.

Run:  python train.py

Trains on the cached features (fast on CPU), keeps the best epoch by validation
macro-AUC, then reports per-class ROC-AUC + macro-F1 on the held-out test set
and saves the model for the demo app.
"""
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                             precision_recall_fscore_support)

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

    history = []
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
        train_loss = running / len(Xtr)
        head.eval()
        with torch.no_grad():
            val_loss = criterion(head(Xval), yval).item()
        _, _, val_auc = evaluate(head, Xval, yval)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "val_macro_auc": val_auc})
        print(f"epoch {epoch:02d}  train_loss {train_loss:.4f}  "
              f"val_loss {val_loss:.4f}  val_macroAUC {val_auc:.4f}")
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.clone() for k, v in head.state_dict().items()}

    # ---- Final test evaluation with the best epoch ----
    head.load_state_dict(best_state)
    probs, aucs, macro_auc = evaluate(head, Xte, yte)
    preds = (probs >= config.DECISION_THRESHOLD).astype(int)
    y_true = yte.numpy()

    # Per-class precision / recall / F1 / support at the decision threshold
    p, r, f1, support = precision_recall_fscore_support(
        y_true, preds, average=None, zero_division=0,
        labels=list(range(config.NUM_CLASSES)))
    macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, preds, average="micro", zero_division=0)
    macro_p, macro_r, _, _ = precision_recall_fscore_support(
        y_true, preds, average="macro", zero_division=0)

    # Multi-label "accuracy" variants (the report explains why these differ)
    subset_acc = accuracy_score(y_true, preds)          # exact match of all 14 labels
    per_label_acc = float((preds == y_true).mean())     # mean correctness per label

    per_class = {}
    for i, name in enumerate(config.DISEASE_LABELS):
        per_class[name] = {
            "auc": float(aucs.get(name, float("nan"))),
            "precision": float(p[i]), "recall": float(r[i]),
            "f1": float(f1[i]), "support": int(support[i]),
        }

    metrics = {
        "backbone": config.BACKBONE,
        "threshold": config.DECISION_THRESHOLD,
        "n_train": len(Xtr), "n_val": len(Xval), "n_test": len(Xte),
        "best_val_macro_auc": float(best_auc),
        "test_macro_auc": float(macro_auc),
        "test_macro_f1": float(macro_f1),
        "test_micro_f1": float(micro_f1),
        "test_macro_precision": float(macro_p),
        "test_macro_recall": float(macro_r),
        "test_subset_accuracy": float(subset_acc),
        "test_per_label_accuracy": per_label_acc,
        "per_class": per_class,
    }

    lines = [
        "=== TEST RESULTS ===",
        f"Backbone                 : {config.BACKBONE}",
        f"Best val macro-AUC       : {best_auc:.4f}",
        f"Test macro-AUC           : {macro_auc:.4f}",
        f"Test macro-F1 (@{config.DECISION_THRESHOLD})   : {macro_f1:.4f}",
        f"Test micro-F1 (@{config.DECISION_THRESHOLD})   : {micro_f1:.4f}",
        f"Test per-label accuracy  : {per_label_acc:.4f}",
        f"Test subset accuracy     : {subset_acc:.4f}",
        "",
        f"{'Class':20s} {'AUC':>6s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'N+':>5s}",
    ]
    for name in sorted(per_class, key=lambda n: -per_class[n]["auc"]):
        c = per_class[name]
        lines.append(f"{name:20s} {c['auc']:6.3f} {c['precision']:6.3f} "
                     f"{c['recall']:6.3f} {c['f1']:6.3f} {c['support']:5d}")
    report = "\n".join(lines)
    print("\n" + report)
    config.METRICS_PATH.write_text(report)
    (config.ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (config.ARTIFACTS_DIR / "history.json").write_text(json.dumps(history, indent=2))

    torch.save({
        "state_dict": head.state_dict(),
        "feat_dim": feat_dim,
        "labels": config.DISEASE_LABELS,
        "backbone": config.BACKBONE,
        "threshold": config.DECISION_THRESHOLD,
    }, config.MODEL_PATH)
    print(f"\nSaved model   -> {config.MODEL_PATH}")
    print(f"Saved metrics -> {config.METRICS_PATH}, metrics.json, history.json")
    print("Next: streamlit run app.py")


if __name__ == "__main__":
    main()
