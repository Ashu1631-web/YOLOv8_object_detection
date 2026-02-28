import streamlit as st
import os
import cv2
import time
from ultralytics import YOLO
from PIL import Image
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="YOLOv8 Pro Detection", layout="wide")

# ---------------- DARK UI ----------------
st.markdown("""
<style>
body {background-color: #0E1117; color: white;}
.stApp {background-color: #0E1117;}
h1, h2, h3 {color: #FFFFFF;}
</style>
""", unsafe_allow_html=True)

st.title("🚀 Real-Time Object Detection using YOLOv8")

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Settings")

model_option = st.sidebar.selectbox(
    "Select Model",
    ["Custom Model (best.pt)", "COCO Model (yolov8n.pt)"]
)

confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model(option):
    if option == "Custom Model (best.pt)":
        return YOLO("best.pt")
    else:
        return YOLO("yolov8n.pt")

model = load_model(model_option)

# ---------------- METRICS ----------------
st.sidebar.subheader("📊 Evaluation Metrics")

metrics_path = "metrics.txt"

if os.path.exists(metrics_path):
    with open(metrics_path, "r") as f:
        metrics_data = f.readlines()
    for line in metrics_data:
        st.sidebar.write(line.strip())
else:
    st.sidebar.warning("metrics.txt not found")

# ---------------- CONFUSION MATRIX ----------------
st.subheader("📈 Confusion Matrix")

if os.path.exists("confusion_matrix.png"):
    st.image("confusion_matrix.png")
elif os.path.exists("outputs/confusion_matrix.png"):
    st.image("outputs/confusion_matrix.png")
else:
    st.warning("Confusion matrix image not found")

# ---------------- DATASET IMAGE + VIDEO TEST ----------------
st.subheader("📂 Test on Dataset Files")

dataset_path = "datasets"

if os.path.exists(dataset_path):

    supported_files = [
        f for f in os.listdir(dataset_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".mp4"))
    ]

    if supported_files:
        # Sort files for clean display
supported_files = sorted(supported_files)

# Create display-friendly names
display_names = {f: f[:40] + "..." if len(f) > 40 else f for f in supported_files}

selected_display = st.selectbox(
    "Select File from Dataset (Search Enabled)",
    list(display_names.values())
)

# Get original filename
selected_file = None
for key, value in display_names.items():
    if value == selected_display:
        selected_file = key
        break

        if selected_file:
            file_path = os.path.join(dataset_path, selected_file)
            file_ext = selected_file.split(".")[-1].lower()

            # -------- IMAGE --------
            if file_ext in ["jpg", "jpeg", "png"]:
                img = cv2.imread(file_path)

                start_time = time.time()
                results = model(img, conf=confidence)
                end_time = time.time()

                fps = 1 / (end_time - start_time)
                annotated = results[0].plot()

                st.image(annotated, channels="BGR")
                st.success(f"FPS: {fps:.2f}")

            # -------- VIDEO --------
            elif file_ext == "mp4":
                cap = cv2.VideoCapture(file_path)
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

    else:
        st.warning("No supported files found in datasets folder.")

else:
    st.warning("datasets folder not found.")

# ---------------- FILE UPLOAD ----------------
st.subheader("📤 Upload Image or Video")

uploaded_file = st.file_uploader(
    "Upload Image or Video",
    type=["jpg", "jpeg", "png", "mp4"]
)

if uploaded_file is not None:
    file_ext = uploaded_file.name.split(".")[-1].lower()

    # -------- IMAGE --------
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

    # -------- VIDEO --------
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
