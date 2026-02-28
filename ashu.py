import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import time
import numpy as np

st.set_page_config(page_title="YOLOv8 Advanced Detection", layout="wide")

st.title("🚀 YOLOv8 Object Detection + Tracking")

# ===============================
# SIDEBAR SETTINGS
# ===============================

st.sidebar.header("⚙️ Settings")

# Load available models
if not os.path.exists("models"):
    st.error("Models folder not found!")
    st.stop()

model_files = [f for f in os.listdir("models") if f.endswith(".pt")]

if len(model_files) == 0:
    st.error("No .pt models found inside models/ folder")
    st.stop()

selected_model = st.sidebar.selectbox("Select Model", model_files)

model_path = os.path.join("models", selected_model)
model = YOLO(model_path)

confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25)

enable_tracking = st.sidebar.checkbox("Enable Tracking")

source_option = st.sidebar.radio("Select Source", ["Image", "Video", "Webcam"])

# Get class names safely
class_names = list(model.names.values())

selected_classes = st.sidebar.multiselect(
    "Filter Classes (optional)",
    class_names,
    default=class_names
)

if selected_classes:
    class_ids = [class_names.index(cls) for cls in selected_classes]
else:
    class_ids = None

# ===============================
# IMAGE DETECTION
# ===============================

if source_option == "Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        start = time.time()

        results = model(image, conf=confidence, classes=class_ids)

        end = time.time()
        fps = 1 / (end - start)

        st.write("Detections:", len(results[0].boxes))

        annotated = results[0].plot()
        st.image(annotated, channels="BGR")

        st.success(f"FPS: {fps:.2f}")

# ===============================
# VIDEO DETECTION
# ===============================

elif source_option == "Video":
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        total_time = 0
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            start = time.time()

            if enable_tracking:
                results = model.track(frame, conf=confidence, persist=True)
            else:
                results = model(frame, conf=confidence, classes=class_ids)

            end = time.time()

            total_time += (end - start)
            frame_count += 1

            annotated = results[0].plot()
            stframe.image(annotated, channels="BGR")

        cap.release()

        if frame_count > 0:
            avg_fps = frame_count / total_time
            st.success(f"Average FPS: {avg_fps:.2f}")

# ===============================
# WEBCAM (Streamlit Compatible)
# ===============================

elif source_option == "Webcam":
    picture = st.camera_input("Capture Frame")

    if picture:
        file_bytes = np.asarray(bytearray(picture.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        start = time.time()

        results = model(frame, conf=confidence, classes=class_ids)

        end = time.time()

        annotated = results[0].plot()
        st.image(annotated, channels="BGR")

        fps = 1 / (end - start)
        st.success(f"FPS: {fps:.2f}")
