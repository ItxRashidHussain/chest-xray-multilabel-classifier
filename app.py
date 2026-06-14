"""STEP 4 - The demo. Upload a chest X-ray, get multi-label predictions.

Run:  streamlit run app.py
"""
import numpy as np
import streamlit as st
import torch
from PIL import Image

import config
from dataset import build_transform
from model import build_backbone, ClassifierHead

st.set_page_config(page_title="Chest X-ray Multi-Label Classifier", layout="centered")
st.title("🫁 Chest X-ray Multi-Label Disease Classifier")
st.caption("Upload a chest X-ray; the model estimates the probability of 14 conditions.")


@st.cache_resource
def load_models():
    if not config.MODEL_PATH.exists():
        return None
    ckpt = torch.load(config.MODEL_PATH, map_location="cpu", weights_only=False)
    backbone, _ = build_backbone(ckpt["backbone"])
    head = ClassifierHead(ckpt["feat_dim"])
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    return {
        "backbone": backbone,
        "head": head,
        "labels": ckpt["labels"],
        "threshold": ckpt["threshold"],
        "transform": build_transform(),
    }


bundle = load_models()
if bundle is None:
    st.error("No trained model found. Run prepare_data.py → extract_features.py → train.py first.")
    st.stop()

uploaded = st.file_uploader("Upload a chest X-ray", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    col1, col2 = st.columns(2)
    col1.image(img, caption="Uploaded X-ray", use_container_width=True)

    with st.spinner("Analyzing..."):
        x = bundle["transform"](img).unsqueeze(0)  # add batch dimension
        with torch.no_grad():
            feat = bundle["backbone"](x)
            probs = torch.sigmoid(bundle["head"](feat)).squeeze(0).numpy()

    labels = bundle["labels"]
    thr = bundle["threshold"]
    order = np.argsort(-probs)  # most confident first

    with col2:
        st.subheader("Result")
        positives = [labels[i] for i in order if probs[i] >= thr]
        if positives:
            st.success("Likely present: " + ", ".join(positives))
        else:
            st.info("No condition above threshold — likely normal.")

    st.subheader("Confidence per condition")
    for i in order:
        st.write(f"**{labels[i]}** — {probs[i] * 100:.1f}%")
        st.progress(float(probs[i]))

    st.caption("⚠️ Educational university project — NOT a medical device. Do not use for diagnosis.")
