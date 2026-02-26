import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np

st.set_page_config(page_title="Hybrid YOLO Detection", layout="wide")
st.title("🚀 Hybrid Detection System")

# Load models
custom_model = YOLO("best.pt")      # Ambulance model
coco_model = YOLO("yolov8n.pt")     # COCO model

confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.6)

mode = st.radio("Select Mode", ["Image", "Video"])


# ---------------- IMAGE MODE ----------------
if mode == "Image":

    file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if file:
        image_bytes = file.read()
        image_np = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        # COCO → person, car, motorcycle, bus, truck
        coco_results = coco_model(
            image,
            conf=confidence,
            classes=[0,2,3,5,7]
        )

        # Custom → ambulance only (class 0 in your dataset)
        custom_results = custom_model(
            image,
            conf=confidence,
            classes=[0]
        )

        # Draw COCO predictions
        frame = coco_results[0].plot()

        # Overlay Custom predictions (ambulance)
        custom_frame = custom_results[0].plot()

        # Combine both detections
        combined = cv2.addWeighted(frame, 1, custom_frame, 1, 0)

        st.image(combined, channels="BGR")


# ---------------- VIDEO MODE ----------------
if mode == "Video":

    file = st.file_uploader("Upload Video", type=["mp4", "avi"])

    if file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            coco_results = coco_model(
                frame,
                conf=confidence,
                classes=[0,2,3,5,7]
            )

            custom_results = custom_model(
                frame,
                conf=confidence,
                classes=[0]
            )

            annotated_coco = coco_results[0].plot()
            annotated_custom = custom_results[0].plot()

            combined = cv2.addWeighted(
                annotated_coco, 1,
                annotated_custom, 1,
                0
            )

            stframe.image(combined, channels="BGR")

        cap.release()