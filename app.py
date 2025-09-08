# app.py (updated)
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json

# Load model
model = tf.keras.models.load_model("garbage_model.h5")

# Load label mapping saved during training
with open("labels.json", "r") as f:
    class_indices = json.load(f)  # e.g. {"batteries":0, "biological":1, ...}
# invert: index -> class_name
inv_map = {int(v): k for k, v in class_indices.items()}

st.set_page_config(page_title="Smart City Garbage Classifier", layout="centered")
st.title("🌆 Smart City Garbage Classification")
st.write("Upload a waste image and AI will classify it for **Smart Recycling**.")

uploaded_file = st.file_uploader("📤 Upload an image of garbage", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess (must match training preprocessing)
    img_resized = image.resize((128, 128))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]  # softmax vector
    top3_idx = preds.argsort()[-3:][::-1]

    st.success(f"♻️ Top prediction: **{inv_map[int(top3_idx[0])]}** ({preds[top3_idx[0]]*100:.2f}%)")

    # Show top-3
    st.markdown("### Top-3 predictions")
    for i, idx in enumerate(top3_idx):
        label = inv_map[int(idx)]
        st.write(f"{i+1}. **{label}** — {preds[idx]*100:.2f}%")

    # Smart City Action (map label to action)
    smart_actions = {
        "plastic": "Send to Plastic Recycling Plant",
        "paper": "Send to Paper Mill",
        "metal": "Send to Metal Recycling Plant",
        "cardboard": "Compress and Reuse",
        "glass_brown": "Glass Recycling",
        "glass_green": "Glass Recycling",
        "glass_white": "Glass Recycling",
        "clothes": "Donate or Textile Recycling",
        "shoes": "Donate or Rubber Recycling",
        "batteries": "Dispose Safely in E-Waste Center",
        "trash": "General Waste Bin",
        "biological": "Convert into Compost"
    }
    # find best action safely
    best_label = inv_map[int(top3_idx[0])]
    st.info(f"🏭 Suggested Action: {smart_actions.get(best_label, 'Check local recycling rules')}")
