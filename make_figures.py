"""Generate every figure used in the project report into results/.

Run AFTER train.py:  python make_figures.py
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import multilabel_confusion_matrix

import config
from model import ClassifierHead

RESULTS = config.ROOT / "results"
RESULTS.mkdir(exist_ok=True)
labels = config.DISEASE_LABELS

# ---------------------------------------------------------------- load everything
df = pd.read_csv(config.SPLITS_CSV)
metrics = json.loads((config.ARTIFACTS_DIR / "metrics.json").read_text())
history = json.loads((config.ARTIFACTS_DIR / "history.json").read_text())

ckpt = torch.load(config.MODEL_PATH, map_location="cpu", weights_only=False)
head = ClassifierHead(ckpt["feat_dim"])
head.load_state_dict(ckpt["state_dict"])
head.eval()

X_test = np.load(config.FEATURES_DIR / "X_test.npy")
y_test = np.load(config.FEATURES_DIR / "y_test.npy")
with torch.no_grad():
    probs = torch.sigmoid(head(torch.from_numpy(X_test).float())).numpy()
preds = (probs >= config.DECISION_THRESHOLD).astype(int)
test_df = df[df["split"] == "test"].reset_index(drop=True)  # aligns with X_test rows


def present(row):
    found = [l for l in labels if row[l] == 1]
    return ", ".join(found) if found else "No Finding"


# ================================================== Figure 1: sample grid
def fig1():
    targets = ["Effusion", "Cardiomegaly", "Atelectasis", "Nodule",
               "Mass", "Pneumothorax", "Infiltration", "Emphysema"]
    fig, axes = plt.subplots(2, 4, figsize=(13, 7))
    for ax, t in zip(axes.ravel(), targets):
        sub = df[df[t] == 1]
        row = sub.iloc[0]
        ax.imshow(Image.open(row["path"]).convert("L"), cmap="gray")
        ax.set_title(present(row), fontsize=8)
        ax.axis("off")
    fig.suptitle("Figure 1: Sample chest X-rays from the NIH ChestX-ray14 sample dataset",
                 y=0.04, fontsize=11)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(RESULTS / "fig1_sample_grid.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ================================================== helpers for diagrams
def box(ax, x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015",
                                linewidth=1.3, edgecolor="#333", facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=16, linewidth=1.3, color="#333"))


# ================================================== Figure 2: pipeline
def fig2():
    fig, ax = plt.subplots(figsize=(14, 3.2))
    ax.set_xlim(0, 14); ax.set_ylim(0, 3); ax.axis("off")
    stages = [
        ("Input\nChest X-ray\n(1024x1024 PNG)", "#E3F2FD"),
        ("Preprocess\nRGB, resize 224x224\nImageNet normalize", "#E8F5E9"),
        ("Frozen MobileNetV2\n(ImageNet pretrained)\nfeature extractor", "#FFF3E0"),
        ("Feature vector\n1280-d\n(cached to disk)", "#FFF3E0"),
        ("Classifier head\nFC512 -> BN -> ReLU\n-> Dropout -> FC14", "#F3E5F5"),
        ("Sigmoid\n14 disease\nprobabilities", "#FFEBEE"),
    ]
    w, h, gap = 2.0, 1.4, 0.25
    x = 0.15
    for i, (txt, fc) in enumerate(stages):
        box(ax, x, 0.8, w, h, txt, fc)
        if i < len(stages) - 1:
            arrow(ax, x + w, 1.5, x + w + gap, 1.5)
        x += w + gap
    ax.set_title("Figure 2: System architecture of the proposed chest X-ray classifier",
                 fontsize=11)
    fig.savefig(RESULTS / "fig2_pipeline.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ================================================== Figure 3: model architecture
def fig3():
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.set_xlim(0, 6); ax.set_ylim(0, 11); ax.axis("off")
    layers = [
        ("Input features  (1280-d)", "#E3F2FD"),
        ("Linear  1280 -> 512", "#F3E5F5"),
        ("BatchNorm1d (512)", "#F3E5F5"),
        ("ReLU", "#F3E5F5"),
        ("Dropout (p = 0.3)", "#F3E5F5"),
        ("Linear  512 -> 14", "#F3E5F5"),
        ("Sigmoid  ->  14 probabilities", "#FFEBEE"),
    ]
    y = 9.5
    for i, (txt, fc) in enumerate(layers):
        box(ax, 1.2, y, 3.6, 0.8, txt, fc)
        if i < len(layers) - 1:
            arrow(ax, 3.0, y, 3.0, y - 0.55)
        y -= 1.35
    ax.text(3.0, 10.6, "Trainable classifier head", ha="center", fontsize=10, weight="bold")
    ax.text(3.0, 0.2, "Backbone: MobileNetV2 features (frozen, not trained)",
            ha="center", fontsize=9, style="italic")
    ax.set_title("Figure 3: Architecture of the trainable classifier head", fontsize=11)
    fig.savefig(RESULTS / "fig3_model.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ================================================== Figures 4 & 5: curves
def fig45():
    ep = [h["epoch"] for h in history]
    tl = [h["train_loss"] for h in history]
    vl = [h["val_loss"] for h in history]
    va = [h["val_macro_auc"] for h in history]
    best = int(np.argmax(va))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ep, tl, "-o", ms=3, label="Train loss")
    ax.plot(ep, vl, "-s", ms=3, label="Validation loss")
    ax.axvline(ep[best], color="gray", ls="--", lw=1, label=f"Best epoch ({ep[best]})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("BCE loss"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Figure 4: Training vs. validation loss per epoch")
    fig.tight_layout(); fig.savefig(RESULTS / "fig4_loss.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ep, va, "-o", ms=3, color="green", label="Validation macro-AUC")
    ax.axvline(ep[best], color="gray", ls="--", lw=1, label=f"Best epoch ({ep[best]})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Macro ROC-AUC"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Figure 5: Validation macro ROC-AUC per epoch")
    fig.tight_layout(); fig.savefig(RESULTS / "fig5_auc.png", dpi=130); plt.close(fig)


# ================================================== Figure 6: confusion matrices
def fig6():
    mcm = multilabel_confusion_matrix(y_test, preds)
    chosen = ["Effusion", "Emphysema", "Infiltration", "Nodule"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, name in zip(axes.ravel(), chosen):
        i = labels.index(name)
        cm = mcm[i].astype(float)
        cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"]); ax.set_yticklabels(["True 0", "True 1"])
        ax.set_title(f"{name} (AUC={metrics['per_class'][name]['auc']:.2f})")
        for r in range(2):
            for c in range(2):
                ax.text(c, r, f"{cmn[r,c]*100:.0f}%\n({int(cm[r,c])})",
                        ha="center", va="center",
                        color="white" if cmn[r, c] > 0.5 else "black", fontsize=9)
    fig.suptitle("Figure 6: Row-normalised per-class confusion matrices (test set)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(RESULTS / "fig6_confusion.png", dpi=130); plt.close(fig)


# ================================================== Figure 7: per-class AUC bar
def fig7():
    pc = metrics["per_class"]
    names = sorted(pc, key=lambda n: pc[n]["auc"])
    vals = [pc[n]["auc"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#C62828" if v < 0.6 else "#F9A825" if v < 0.7 else "#2E7D32" for v in vals]
    ax.barh(names, vals, color=colors)
    ax.axvline(0.5, color="gray", ls="--", lw=1, label="Random (0.50)")
    ax.axvline(metrics["test_macro_auc"], color="blue", ls=":", lw=1.5,
               label=f"Macro avg ({metrics['test_macro_auc']:.2f})")
    ax.set_xlabel("ROC-AUC"); ax.set_xlim(0.4, 0.9); ax.legend()
    ax.set_title("Figure 7: Per-class ROC-AUC on the test set")
    fig.tight_layout(); fig.savefig(RESULTS / "fig7_per_class_auc.png", dpi=130); plt.close(fig)


# ================================================== Figure 8: sample predictions
def fig8():
    # rank test items: "good" = a true disease predicted with high prob; "poor" otherwise
    scores = []
    for i in range(len(test_df)):
        true_idx = np.where(y_test[i] == 1)[0]
        top = int(np.argmax(probs[i]))
        good = top in true_idx and probs[i][top] >= config.DECISION_THRESHOLD
        scores.append((good, probs[i][top], i))
    good = [s for s in scores if s[0]][:4]
    poor = [s for s in scores if not s[0]][:4]
    picks = good + poor

    fig, axes = plt.subplots(2, 4, figsize=(14, 7.5))
    for ax, (is_good, _, i) in zip(axes.ravel(), picks):
        row = test_df.iloc[i]
        ax.imshow(Image.open(row["path"]).convert("L"), cmap="gray")
        order = np.argsort(-probs[i])[:2]
        pred_txt = ", ".join(f"{labels[j]} {probs[i][j]*100:.0f}%" for j in order)
        true_txt = present(row)
        tag = "OK" if is_good else "MISS"
        ax.set_title(f"[{tag}]\nPred: {pred_txt}\nTrue: {true_txt}", fontsize=7.5,
                     color="green" if is_good else "red")
        ax.axis("off")
    fig.suptitle("Figure 8: Sample test predictions (top: confident hits, bottom: errors)",
                 y=0.02, fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(RESULTS / "fig8_predictions.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1(); print("fig1 sample grid")
    fig2(); print("fig2 pipeline")
    fig3(); print("fig3 model")
    fig45(); print("fig4/5 curves")
    fig6(); print("fig6 confusion")
    fig7(); print("fig7 per-class auc")
    fig8(); print("fig8 predictions")
    print(f"All figures saved to {RESULTS}")
