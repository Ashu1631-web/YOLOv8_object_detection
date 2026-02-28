import streamlit as st
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
from ultralytics import YOLO

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="YOLO Object Detection", layout="wide")

st.markdown("""
<h2 style='text-align:center;color:#1F77B4;'>
🚀 YOLO Object Detection - Fast YOLOv8 Dashboard
</h2>
<hr>
""", unsafe_allow_html=True)

# -------------------------------------------------
# MODEL LOADER (TRUE LAZY LOAD)
# -------------------------------------------------
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

def get_model(model_option):
    if (
        "model" not in st.session_state
        or st.session_state.model_name != model_option
    ):
        with st.spinner("Loading Model..."):
            st.session_state.model = load_model(model_option)
            st.session_state.model_name = model_option
    return st.session_state.model

# -------------------------------------------------
# SIDEBAR SETTINGS
# -------------------------------------------------
st.sidebar.header("⚙ Settings")

model_option = st.sidebar.selectbox(
    "Select Model",
    ["yolov8n.pt", "yolov8s.pt", "best.pt"]
)

confidence = st.sidebar.slider("Confidence", 0.1, 1.0, 0.30)
iou = st.sidebar.slider("IoU Threshold", 0.1, 1.0, 0.45)
imgsz = st.sidebar.selectbox("Image Size", [320, 480, 640], index=2)

# -------------------------------------------------
# TABS
# -------------------------------------------------
tab1, tab2 = st.tabs(["🔍 Detection", "📊 Analytics"])

# -------------------------------------------------
# DETECTION TAB
# -------------------------------------------------
with tab1:

    mode = st.radio(
        "Select Mode",
        ["Upload Image", "Dataset Image", "Video"]
    )

    # ---------------- UPLOAD IMAGE ----------------
    if mode == "Upload Image":

        uploaded_image = st.file_uploader(
            "Upload Image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_image:

            model = get_model(model_option)

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

            st.session_state.last_results = results

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

    # ---------------- DATASET IMAGE ----------------
    if mode == "Dataset Image":

        dataset_path = "datasets"

        if os.path.exists(dataset_path):

            image_files = [
                f for f in os.listdir(dataset_path)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
            ]

            if image_files:

                selected_image = st.selectbox(
                    "Select Dataset Image",
                    image_files
                )

                if selected_image:

                    model = get_model(model_option)

                    image_path = os.path.join(
                        dataset_path,
                        selected_image
                    )

                    image = cv2.imread(image_path)

                    results = model.predict(
                        source=image,
                        conf=confidence,
                        iou=iou,
                        imgsz=imgsz,
                        device="cpu"
                    )

                    st.session_state.last_results = results

                    annotated = results[0].plot()
                    annotated = cv2.cvtColor(
                        annotated,
                        cv2.COLOR_BGR2RGB
                    )

                    st.image(
                        annotated,
                        use_column_width=True
                    )
            else:
                st.warning("No images found in datasets folder.")
        else:
            st.warning("datasets folder not found.")

    # ---------------- VIDEO ----------------
    if mode == "Video":

        uploaded_video = st.file_uploader(
            "Upload Video",
            type=["mp4", "avi", "mov"]
        )

        if uploaded_video:

            model = get_model(model_option)

            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())

            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()

            all_class_ids = []
            all_conf_scores = []

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

                boxes = results[0].boxes

                if len(boxes) > 0:
                    all_class_ids.extend(
                        boxes.cls.cpu().numpy()
                    )
                    all_conf_scores.extend(
                        boxes.conf.cpu().numpy()
                    )

                annotated = results[0].plot()
                annotated = cv2.cvtColor(
                    annotated,
                    cv2.COLOR_BGR2RGB
                )

                stframe.image(annotated)

            cap.release()

            st.session_state.video_class_ids = all_class_ids
            st.session_state.video_conf_scores = all_conf_scores

            st.success("Video Processing Completed")

# -------------------------------------------------
# ANALYTICS TAB
# -------------------------------------------------
with tab2:

    # Image Analytics
    if "last_results" in st.session_state:

        boxes = st.session_state.last_results[0].boxes

        if len(boxes) > 0:

            class_ids = boxes.cls.cpu().numpy()
            conf_scores = boxes.conf.cpu().numpy()

            st.subheader("📊 Image Class Distribution")
            st.bar_chart(
                pd.Series(class_ids).value_counts()
            )

            st.subheader("📈 Image Confidence Distribution")

            fig, ax = plt.subplots()
            ax.hist(conf_scores, bins=10)
            ax.set_xlabel("Confidence")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

    # Video Analytics
    if "video_class_ids" in st.session_state:

        st.subheader("🎬 Video Class Distribution")
        st.bar_chart(
            pd.Series(
                st.session_state.video_class_ids
            ).value_counts()
        )

        st.subheader("🎬 Video Confidence Distribution")

        fig2, ax2 = plt.subplots()
        ax2.hist(
            st.session_state.video_conf_scores,
            bins=10
        )
        ax2.set_xlabel("Confidence")
        ax2.set_ylabel("Frequency")
        st.pyplot(fig2)
