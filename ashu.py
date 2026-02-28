import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import time
import os
import numpy as np

st.set_page_config(page_title="Advanced YOLOv8 Detection", layout="wide")

st.title("🚀 Advanced YOLOv8 Object Detection + Tracking")

# Load Model
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

# Sidebar Controls
st.sidebar.header("⚙️ Settings")

confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3)
enable_tracking = st.sidebar.checkbox("Enable Tracking", value=False)

class_names = list(model.names.values())
selected_classes = st.sidebar.multiselect(
    "Select Classes (optional)",
    class_names,
    default=class_names
)

source_option = st.sidebar.radio(
    "Choose Source",
    ["Image", "Video", "Webcam"]
)

output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------
# IMAGE MODE
# ---------------------------
if source_option == "Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        start = time.time()

        results = model(
            image,
            conf=confidence,
            classes=[class_names.index(cls) for cls in selected_classes]
        )

        end = time.time()
        fps = 1 / (end - start)

        annotated_frame = results[0].plot()

        st.image(annotated_frame, channels="BGR")
        st.success(f"FPS: {fps:.2f}")

        save_path = os.path.join(output_dir, "output_image.jpg")
        cv2.imwrite(save_path, annotated_frame)

# ---------------------------
# VIDEO MODE
# ---------------------------
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
                results = model.track(
                    frame,
                    conf=confidence,
                    persist=True
                )
            else:
                results = model(
                    frame,
                    conf=confidence,
                    classes=[class_names.index(cls) for cls in selected_classes]
                )

            end = time.time()

            total_time += (end - start)
            frame_count += 1

            annotated_frame = results[0].plot()
            stframe.image(annotated_frame, channels="BGR")

        cap.release()

        if frame_count > 0:
            avg_fps = frame_count / total_time
            st.success(f"Average FPS: {avg_fps:.2f}")

# ---------------------------
# WEBCAM MODE
# ---------------------------
elif source_option == "Webcam":
    run = st.checkbox("Start Webcam")

    if run:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        total_time = 0
        frame_count = 0

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam")
                break

            start = time.time()

            if enable_tracking:
                results = model.track(
                    frame,
                    conf=confidence,
                    persist=True
                )
            else:
                results = model(
                    frame,
                    conf=confidence,
                    classes=[class_names.index(cls) for cls in selected_classes]
                )

            end = time.time()

            total_time += (end - start)
            frame_count += 1

            annotated_frame = results[0].plot()
            stframe.image(annotated_frame, channels="BGR")

        cap.release()

        if frame_count > 0:
            avg_fps = frame_count / total_time
            st.success(f"Average FPS: {avg_fps:.2f}")
