import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os
import subprocess
from collections import Counter

st.set_page_config(page_title="Advanced YOLOv8 Dashboard", layout="wide")

# -------------------- MODEL CACHE --------------------
@st.cache_resource
def load_model(path):
    return YOLO(path)

# -------------------- SIDEBAR --------------------
st.sidebar.title("⚙ Control Panel")

task = st.sidebar.radio("Select Mode", ["Image", "Video"])
model_choice = st.sidebar.selectbox("Select Model", ["yolov8n.pt"])
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)
iou = st.sidebar.slider("IOU Threshold", 0.1, 1.0, 0.45)

model = load_model(model_choice)

# Class Filter
class_names = list(model.names.values())
selected_classes = st.sidebar.multiselect(
    "Filter Classes (Optional)", class_names, default=class_names
)

# -------------------- TABS --------------------
tab1, tab2 = st.tabs(["🚀 Detection", "📊 Model Evaluation"])

# ================== DETECTION TAB ==================
with tab1:
    st.title("🚀 Real-Time Object Detection using YOLOv8")

    # ---------------- IMAGE MODE ----------------
    if task == "Image":
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            image = Image.open(uploaded_file)
            img_array = np.array(image)

            col1, col2 = st.columns(2)

            with col1:
                st.image(image, caption="Original Image", use_column_width=True)

            with st.spinner("Detecting objects..."):
                results = model(img_array, conf=confidence, iou=iou)

            # Filter classes
            filtered_boxes = []
            for box in results[0].boxes:
                cls = model.names[int(box.cls)]
                if cls in selected_classes:
                    filtered_boxes.append(box)

            results[0].boxes = filtered_boxes
            result_img = results[0].plot()

            with col2:
                st.image(result_img, caption="Detected Image", use_column_width=True)

            # ---------------- ANALYTICS ----------------
            counts = Counter(
                model.names[int(box.cls)] for box in filtered_boxes
            )

            st.divider()
            colA, colB = st.columns(2)

            colA.metric("Total Objects", sum(counts.values()))
            if counts:
                avg_conf = np.mean([float(box.conf) for box in filtered_boxes])
                colB.metric("Avg Confidence", round(avg_conf, 2))

            st.subheader("Class-wise Count")
            st.table(counts)

            # Save result for download
            output_path = "detected_image.jpg"
            cv2.imwrite(output_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))

            with open(output_path, "rb") as file:
                st.download_button(
                    "⬇ Download Result",
                    file,
                    file_name="detected_image.jpg"
                )

    # ---------------- VIDEO MODE ----------------
    elif task == "Video":
        uploaded_video = st.file_uploader("Upload Video", type=["mp4"])

        if uploaded_video:
            temp_video = tempfile.NamedTemporaryFile(delete=False)
            temp_video.write(uploaded_video.read())

            cap = cv2.VideoCapture(temp_video.name)

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            output_path = "output_video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress = st.progress(0)

            frame_count = 0

            with st.spinner("Processing video..."):
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model(frame, conf=confidence, iou=iou)
                    result_frame = results[0].plot()
                    out.write(result_frame)

                    frame_count += 1
                    progress.progress(frame_count / total_frames)

            cap.release()
            out.release()

            st.success("Video Processing Complete")

            # Optional compression using FFmpeg
            compressed_output = "compressed_output.mp4"
            subprocess.run([
                "ffmpeg",
                "-i", output_path,
                "-vcodec", "libx264",
                "-crf", "28",
                "-preset", "fast",
                "-movflags", "+faststart",
                compressed_output
            ])

            st.video(compressed_output)

            with open(compressed_output, "rb") as file:
                st.download_button(
                    "⬇ Download Processed Video",
                    file,
                    file_name="detected_video.mp4"
                )

# ================== EVALUATION TAB ==================
with tab2:
    st.title("📊 Model Evaluation")

    st.info("Add your confusion matrix and metrics visualization here.")

    st.metric("mAP50", 0.57)
    st.metric("mAP50-95", 0.40)
    st.metric("Precision", 0.66)
    st.metric("Recall", 0.53)
