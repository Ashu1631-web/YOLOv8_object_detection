import streamlit as st
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
from ultralytics import YOLO

# -------------------------------------------------
# PAGE CONFIG + BRANDING
# -------------------------------------------------
st.set_page_config(page_title="Ashish AI Vision", layout="wide")

st.markdown("""
<h1 style='text-align: center; color: #1F77B4;'>
🚀 Ashish AI Vision - YOLOv8 Detection Suite
</h1>
<hr>
""", unsafe_allow_html=True)

# -------------------------------------------------
# MODEL LOADING (CACHED FOR PERFORMANCE)
# -------------------------------------------------
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# -------------------------------------------------
# SIDEBAR SETTINGS
# -------------------------------------------------
st.sidebar.header("⚙ Detection Settings")

model_option = st.sidebar.selectbox(
    "Select Model",
    ["yolov8s.pt", "yolov8n.pt", "best.pt"]
)

confidence = st.sidebar.slider("Confidence", 0.1, 1.0, 0.30)
iou = st.sidebar.slider("IoU Threshold", 0.1, 1.0, 0.45)
imgsz = st.sidebar.selectbox("Image Size", [320, 480, 640], index=2)

model = load_model(model_option)

# Class Filtering
class_filter = st.sidebar.multiselect(
    "Filter Classes (Optional)",
    list(model.names.values())
)

selected_class_ids = None
if class_filter:
    selected_class_ids = [
        k for k, v in model.names.items()
        if v in class_filter
    ]

# -------------------------------------------------
# TABS
# -------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    ["🔍 Detection", "📊 Analytics", "🧪 Validation"]
)

# -------------------------------------------------
# DETECTION TAB
# -------------------------------------------------
with tab1:

    mode = st.radio("Select Mode", ["Image", "Video"])

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
                classes=selected_class_ids,
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
                st.metric("🎯 Confidence", confidence)

            # Download button
            _, buffer = cv2.imencode(
                ".jpg",
                cv2.cvtColor(
                    annotated,
                    cv2.COLOR_RGB2BGR
                )
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

            fps_list = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                start = time.time()

                results = model.predict(
                    source=frame,
                    conf=confidence,
                    iou=iou,
                    imgsz=imgsz,
                    classes=selected_class_ids,
                    device="cpu"
                )

                end = time.time()
                fps = 1 / (end - start)
                fps_list.append(fps)

                annotated = results[0].plot()
                annotated = cv2.cvtColor(
                    annotated,
                    cv2.COLOR_BGR2RGB
                )

                stframe.image(annotated)

            cap.release()
            st.success("Video Processing Completed")

# -------------------------------------------------
# ANALYTICS TAB
# -------------------------------------------------
with tab2:

    if "results" in locals():

        boxes = results[0].boxes

        if len(boxes) > 0:

            class_ids = boxes.cls.cpu().numpy()
            conf_scores = boxes.conf.cpu().numpy()

            st.subheader("📌 Class Distribution")
            st.bar_chart(
                pd.Series(class_ids).value_counts()
            )

            st.subheader("📊 Confidence Distribution")
            fig, ax = plt.subplots()
            ax.hist(conf_scores, bins=10)
            ax.set_xlabel("Confidence Score")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

        else:
            st.info("No detections available.")

# -------------------------------------------------
# VALIDATION TAB
# -------------------------------------------------
with tab3:

    st.subheader("Model Validation")

    dataset_yaml = st.text_input(
        "Enter dataset YAML path (example: datasets/data.yaml)"
    )

    if st.button("Run Validation") and dataset_yaml:

        with st.spinner("Validating Model..."):

            metrics = model.val(data=dataset_yaml)

            st.write("### Evaluation Metrics")
            st.write(metrics.results_dict)

            if hasattr(metrics, "confusion_matrix"):
                cm = metrics.confusion_matrix.matrix

                fig2, ax2 = plt.subplots(figsize=(8, 6))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues"
                )
                ax2.set_title("Confusion Matrix")
                st.pyplot(fig2)

st.markdown("""
<hr>
<center>
Built with ❤️ by Ashish | YOLOv8 Powered
</center>
""", unsafe_allow_html=True)
