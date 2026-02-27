import streamlit as st
from ultralytics import YOLO
import tempfile
import os
import torch
from PIL import Image
import numpy as np

# Force CPU (important for Streamlit Cloud)
device = "cpu"

st.set_page_config(page_title="YOLOv8 Object Detection", layout="wide")

st.title("🚀 YOLOv8 Object Detection System")

# Model selection
model_option = st.selectbox(
    "Select Model",
    ["COCO Model (yolov8n.pt)", "Custom Model (best.pt)"]
)

# Load model safely
@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path)
    model.to(device)
    return model

if model_option == "COCO Model (yolov8n.pt)":
    model = load_model("yolov8n.pt")
else:
    model = load_model("best.pt")

confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    try:
        # Run detection
        results = model(tmp_path, conf=confidence)

        # Get annotated image
        annotated_frame = results[0].plot()

        # Convert to RGB (Streamlit compatibility)
        annotated_frame = Image.fromarray(annotated_frame[..., ::-1])

        st.image(annotated_frame, caption="Detection Result", use_column_width=True)

    except Exception as e:
        st.error(f"Error during detection: {e}")

    finally:
        os.remove(tmp_path)
