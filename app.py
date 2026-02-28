import streamlit as st
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

st.set_page_config(page_title="YOLOv8 Detection", layout="wide")

# -------------------------------
# Load Model (Cached)
# -------------------------------
@st.cache_resource
def load_model(path):
    return YOLO(path)

# -------------------------------
# Sidebar Controls
# -------------------------------
st.sidebar.title("⚙ Settings")

model_option = st.sidebar.selectbox(
    "Select Model",
    ["yolov8n.pt", "yolov8s.pt", "best.pt"]
)

confidence = st.sidebar.slider("Confidence", 0.1, 1.0, 0.25)
iou = st.sidebar.slider("IoU Threshold", 0.1, 1.0, 0.45)
imgsz = st.sidebar.selectbox("Image Size", [320, 480, 640])

model = load_model(model_option)

# -------------------------------
# Main Layout
# -------------------------------
tab1, tab2 = st.tabs(["🔍 Detection", "📊 Analytics"])

# -------------------------------
# Detection Tab
# -------------------------------
with tab1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        start_time = time.time()

        results = model.predict(
            source=image,
            conf=confidence,
            iou=iou,
            imgsz=imgsz,
            device="cpu"
        )

        end_time = time.time()
        fps = 1 / (end_time - start_time)

        annotated = results[0].plot()
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns([2,1])

        with col1:
            st.image(annotated, use_column_width=True)

        with col2:
            st.metric("FPS", f"{fps:.2f}")
            st.metric("Objects", len(results[0].boxes))
            st.metric("Confidence", f"{confidence}")

# -------------------------------
# Analytics Tab
# -------------------------------
with tab2:
    if uploaded_file:
        boxes = results[0].boxes

        if len(boxes) > 0:
            class_ids = boxes.cls.cpu().numpy()
            df = pd.Series(class_ids).value_counts()

            st.subheader("📌 Class Distribution")
            st.bar_chart(df)

            # Confidence Distribution
            conf_scores = boxes.conf.cpu().numpy()
            fig, ax = plt.subplots()
            ax.hist(conf_scores, bins=10)
            ax.set_title("Confidence Distribution")
            ax.set_xlabel("Confidence")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
        else:
            st.info("No detections to analyze.")
