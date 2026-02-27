from ultralytics import YOLO
import streamlit as st
from PIL import Image
import tempfile
import os

st.title("🚀 YOLOv8 Object Detection System")

model_option = st.selectbox(
    "Select Model",
    ["COCO Model (yolov8n.pt)", "Custom Model (best.pt)"]
)

confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)

if model_option == "COCO Model (yolov8n.pt)":
    model = YOLO("yolov8n.pt")
else:
    model = YOLO("best.pt")

uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Run detection
    results = model(tmp_path, conf=confidence)

    # Plot results
    annotated_frame = results[0].plot()

    st.image(annotated_frame, caption="Detection Result", use_column_width=True)

    # Cleanup temp file
    os.remove(tmp_path)
