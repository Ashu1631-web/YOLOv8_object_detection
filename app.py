import streamlit as st
import os
import cv2
import time
from ultralytics import YOLO
from PIL import Image
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="YOLOv8 Pro Detection", layout="wide")

# ---------------- DARK THEME CSS ----------------
st.markdown("""
<style>
body {background-color: #0E1117; color: white;}
.stApp {background-color: #0E1117;}
</style>
""", unsafe_allow_html=True)

st.title("🚀 Real-Time Object Detection using YOLOv8")

# ---------------- MODEL SELECTION ----------------
model_option = st.sidebar.selectbox(
    "Select Model",
    ["Custom Model (best.pt)", "COCO Model (yolov8n.pt)"]
)

if model_option == "Custom Model (best.pt)":
    model = YOLO("best.pt")
else:
    model = YOLO("yolov8n.pt")

confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3)

# ---------------- METRICS LOAD ----------------
st.sidebar.subheader("📊 Evaluation Metrics")

metrics_path = "metrics.txt"

if os.path.exists(metrics_path):
    with open(metrics_path, "r") as f:
        metrics_data = f.readlines()

    for line in metrics_data:
        st.sidebar.write(line.strip())
else:
    st.sidebar.warning("Metrics file not found")

# ---------------- CONFUSION MATRIX ----------------
st.subheader("📈 Confusion Matrix")

if os.path.exists("outputs/confusion_matrix.png"):
    st.image("outputs/confusion_matrix.png")
else:
    st.warning("Confusion matrix image not found")

# ---------------- DATASET IMAGE SELECT ----------------
st.subheader("📂 Test on Dataset Images")

dataset_path = "dataset/images"

if os.path.exists(dataset_path):
    image_files = os.listdir(dataset_path)
    selected_image = st.selectbox("Select Image", image_files)

    if selected_image:
        img_path = os.path.join(dataset_path, selected_image)
        img = cv2.imread(img_path)

        start_time = time.time()
        results = model(img, conf=confidence)
        end_time = time.time()

        fps = 1 / (end_time - start_time)

        annotated = results[0].plot()

        st.image(annotated, channels="BGR")
        st.success(f"FPS: {fps:.2f}")
else:
    st.warning("Dataset folder not found.")

# ---------------- FILE UPLOAD ----------------
st.subheader("📤 Upload Image or Video")

uploaded_file = st.file_uploader(
    "Upload Image or Video",
    type=["jpg", "jpeg", "png", "mp4"]
)

if uploaded_file is not None:
    file_ext = uploaded_file.name.split(".")[-1]

    if file_ext in ["jpg", "jpeg", "png"]:
        image = Image.open(uploaded_file)
        img_array = np.array(image)

        start_time = time.time()
        results = model(img_array, conf=confidence)
        end_time = time.time()

        fps = 1 / (end_time - start_time)

        annotated = results[0].plot()

        st.image(annotated, channels="BGR")
        st.success(f"FPS: {fps:.2f}")

    elif file_ext == "mp4":
        temp_video = "temp.mp4"
        with open(temp_video, "wb") as f:
            f.write(uploaded_file.read())

        cap = cv2.VideoCapture(temp_video)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            start_time = time.time()
            results = model(frame, conf=confidence)
            end_time = time.time()

            fps = 1 / (end_time - start_time)

            annotated = results[0].plot()
            stframe.image(annotated, channels="BGR")

        cap.release()
