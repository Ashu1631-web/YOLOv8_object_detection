import streamlit as st
from ultralytics import YOLO
import tempfile
import os
import torch
import cv2
from PIL import Image
import numpy as np

device = "cpu"

st.set_page_config(page_title="YOLOv8 Object Detection", layout="wide")
st.title("🚀 YOLOv8 Object Detection System")

model_option = st.selectbox(
    "Select Model",
    ["COCO Model (yolov8n.pt)", "Custom Model (best.pt)"]
)

@st.cache_resource
def load_model(path):
    model = YOLO(path)
    model.to(device)
    return model

if model_option == "COCO Model (yolov8n.pt)":
    model = load_model("yolov8n.pt")
else:
    model = load_model("best.pt")

confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.3)

uploaded_file = st.file_uploader(
    "Upload Image or Video",
    type=["jpg", "jpeg", "png", "mp4", "avi"]
)

if uploaded_file is not None:

    file_ext = uploaded_file.name.split(".")[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    try:
        if file_ext in ["jpg", "jpeg", "png"]:
            # IMAGE DETECTION
            results = model(tmp_path, conf=confidence)
            annotated = results[0].plot()
            annotated = Image.fromarray(annotated[..., ::-1])
            st.image(annotated, caption="Detection Result", use_column_width=True)

        elif file_ext == "mp4":
            # VIDEO DETECTION
            cap = cv2.VideoCapture(tmp_path)
            stframe = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame, conf=confidence)
                annotated = results[0].plot()
                stframe.image(annotated, channels="BGR")

            cap.release()

    except Exception as e:
        st.error(f"Error: {e}")

    finally:
        os.remove(tmp_path)
