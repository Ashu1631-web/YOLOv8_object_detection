import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile

st.set_page_config(page_title="YOLOv8 Detection App", layout="wide")

st.title("🚀 YOLOv8 Object Detection System")
st.write("Upload Image or Video for Detection")

# Model selection
model_option = st.selectbox(
    "Select Model",
    ["Custom Model (best.pt)", "COCO Model (yolov8n.pt)"]
)

if model_option == "Custom Model (best.pt)":
    model = YOLO("models/best.pt")
else:
    model = YOLO("models/yolov8n.pt")

confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.5)

uploaded_file = st.file_uploader(
    "Upload Image or Video",
    type=["jpg", "jpeg", "png", "mp4"]
)

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    # IMAGE DETECTION
    if uploaded_file.type.startswith("image"):
        results = model(tfile.name, conf=confidence)
        annotated = results[0].plot()
        st.image(annotated, channels="BGR")

    # VIDEO DETECTION
    else:
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=confidence)
            annotated = results[0].plot()
            stframe.image(annotated, channels="BGR")

        cap.release()

st.markdown("---")
st.markdown("Developed using YOLOv8 + Streamlit")