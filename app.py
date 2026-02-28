import streamlit as st
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(page_title="YOLOv8 Object Detection", layout="wide")

st.title("🚀 YOLOv8 Advanced Detection Dashboard")

# --------------------------------------------------
# Load Model (Cached - Performance Boost)
# --------------------------------------------------
@st.cache_resource
def load_model(path):
    return YOLO(path)

# --------------------------------------------------
# Sidebar Controls
# --------------------------------------------------
st.sidebar.header("⚙ Detection Settings")

model_option = st.sidebar.selectbox(
    "Select Model",
    ["yolov8n.pt", "yolov8s.pt", "best.pt"]
)

confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25)
iou = st.sidebar.slider("IoU Threshold", 0.1, 1.0, 0.45)
imgsz = st.sidebar.selectbox("Image Size", [320, 480, 640])

model = load_model(model_option)

# --------------------------------------------------
# Tabs Layout
# --------------------------------------------------
tab1, tab2 = st.tabs(["🔍 Detection", "📊 Advanced Analytics"])

# --------------------------------------------------
# Detection Tab
# --------------------------------------------------
with tab1:

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:

        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # Start timer
        start_time = time.time()

        results = model.predict(
            source=image,
            conf=confidence,
            iou=iou,
            imgsz=imgsz,
            device="cpu"
        )

        # End timer
        end_time = time.time()
        fps = 1 / (end_time - start_time)

        # Annotated Image
        annotated_frame = results[0].plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.image(annotated_frame, use_column_width=True)

        with col2:
            st.metric("⚡ FPS", f"{fps:.2f}")
            st.metric("📦 Objects Detected", len(results[0].boxes))
            st.metric("🎯 Confidence", f"{confidence}")

        # Download Button
        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
        st.download_button(
            label="📥 Download Result Image",
            data=buffer.tobytes(),
            file_name="detection_result.jpg",
            mime="image/jpeg"
        )

# --------------------------------------------------
# Advanced Analytics Tab
# --------------------------------------------------
with tab2:

    if uploaded_file:

        boxes = results[0].boxes

        if len(boxes) > 0:

            class_ids = boxes.cls.cpu().numpy()
            conf_scores = boxes.conf.cpu().numpy()

            # -----------------------------------
            # Class Distribution
            # -----------------------------------
            st.subheader("📌 Class Distribution")

            class_counts = pd.Series(class_ids).value_counts()
            st.bar_chart(class_counts)

            # -----------------------------------
            # Confidence Histogram
            # -----------------------------------
            st.subheader("📊 Confidence Distribution")

            fig1, ax1 = plt.subplots()
            ax1.hist(conf_scores, bins=10)
            ax1.set_xlabel("Confidence Score")
            ax1.set_ylabel("Frequency")
            st.pyplot(fig1)

            # -----------------------------------
            # Real-Time FPS Trend
            # -----------------------------------
            st.subheader("⚡ FPS Trend")

            if "fps_history" not in st.session_state:
                st.session_state.fps_history = []

            st.session_state.fps_history.append(fps)
            st.line_chart(st.session_state.fps_history)

        else:
            st.info("No detections available for analytics.")
