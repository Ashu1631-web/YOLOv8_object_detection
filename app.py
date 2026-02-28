import streamlit as st
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
from ultralytics import YOLO

# --------------------------------------------
# PAGE CONFIG
# --------------------------------------------
st.set_page_config(page_title="Ashish AI Vision", layout="wide")

st.markdown("""
<h2 style='text-align:center;color:#1F77B4;'>
🚀 Ashish AI Vision - Fast YOLOv8 Dashboard
</h2>
<hr>
""", unsafe_allow_html=True)

# --------------------------------------------
# LAZY MODEL LOAD FUNCTION
# --------------------------------------------
@st.cache_resource
def load_model(path):
    return YOLO(path)

# --------------------------------------------
# SIDEBAR
# --------------------------------------------
st.sidebar.header("⚙ Settings")

model_option = st.sidebar.selectbox(
    "Model",
    ["yolov8s.pt", "best.pt", "yolov8n.pt"]
)

confidence = st.sidebar.slider("Confidence", 0.1, 1.0, 0.30)
iou = st.sidebar.slider("IoU", 0.1, 1.0, 0.45)
imgsz = st.sidebar.selectbox("Image Size", [320, 480, 640], index=2)

# --------------------------------------------
# TABS
# --------------------------------------------
tab1, tab2 = st.tabs(["🔍 Detection", "📊 Analytics"])

# --------------------------------------------
# DETECTION TAB
# --------------------------------------------
with tab1:

    mode = st.radio("Mode", ["Image", "Video"])

    # Lazy load model only when needed
    if "model" not in st.session_state:
        st.session_state.model = load_model(model_option)

    model = st.session_state.model

    # ---------------- IMAGE ----------------
    if mode == "Image":

        uploaded_image = st.file_uploader(
            "Upload Image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_image:

            file_bytes = np.asarray(
                bytearray(uploaded_image.read()),
                dtype=np.uint8
            )
            image = cv2.imdecode(file_bytes, 1)

            start = time.time()

            results = model.predict(
                source=image,
                conf=confidence,
                iou=iou,
                imgsz=imgsz,
                device="cpu"
            )

            end = time.time()
            fps = 1 / (end - start)

            annotated = results[0].plot()
            annotated = cv2.cvtColor(
                annotated,
                cv2.COLOR_BGR2RGB
            )

            col1, col2 = st.columns([2, 1])

            with col1:
                st.image(annotated, use_column_width=True)

            with col2:
                st.metric("⚡ FPS", f"{fps:.2f}")
                st.metric("📦 Objects", len(results[0].boxes))

            # Download result
            _, buffer = cv2.imencode(
                ".jpg",
                cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            )

            st.download_button(
                "📥 Download Result",
                buffer.tobytes(),
                "result.jpg",
                "image/jpeg"
            )

    # ---------------- VIDEO ----------------
    if mode == "Video":

        uploaded_video = st.file_uploader(
            "Upload Video",
            type=["mp4", "avi", "mov"]
        )

        if uploaded_video:

            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())

            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.predict(
                    source=frame,
                    conf=confidence,
                    iou=iou,
                    imgsz=imgsz,
                    device="cpu"
                )

                annotated = results[0].plot()
                annotated = cv2.cvtColor(
                    annotated,
                    cv2.COLOR_BGR2RGB
                )

                stframe.image(annotated)

            cap.release()
            st.success("Video Processing Completed")

# --------------------------------------------
# ANALYTICS TAB (Lightweight)
# --------------------------------------------
with tab2:

    if "results" in locals():

        boxes = results[0].boxes

        if len(boxes) > 0:

            class_ids = boxes.cls.cpu().numpy()
            conf_scores = boxes.conf.cpu().numpy()

            st.subheader("Class Distribution")
            st.bar_chart(
                pd.Series(class_ids).value_counts()
            )

            st.subheader("Confidence Distribution")
            fig, ax = plt.subplots()
            ax.hist(conf_scores, bins=10)
            st.pyplot(fig)

        else:
            st.info("No detections yet.")
