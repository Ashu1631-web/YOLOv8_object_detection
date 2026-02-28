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
st.set_page_config(page_title="YOLOv8 Pro Dashboard", layout="wide")

st.markdown("""
<h2 style='text-align:center;color:#1F77B4;'>
🚀 YOLOv8 Professional Detection Dashboard
</h2>
<hr>
""", unsafe_allow_html=True)

# -------------------------------------------------
# MODEL LOADER
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
# SIDEBAR
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
tab1, tab2, tab3 = st.tabs(["🔍 Detection", "📊 Analytics", "🧪 Dataset & Metrics"])

# -------------------------------------------------
# DETECTION TAB
# -------------------------------------------------
with tab1:

    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_image:
        model = get_model(model_option)

        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
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
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns([2,1])
        with col1:
            st.image(annotated, use_column_width=True)
        with col2:
            st.metric("FPS", f"{fps:.2f}")
            st.metric("Objects", len(results[0].boxes))

# -------------------------------------------------
# ANALYTICS TAB
# -------------------------------------------------
with tab2:

    if "last_results" in st.session_state:
        boxes = st.session_state.last_results[0].boxes

        if len(boxes) > 0:
            class_ids = boxes.cls.cpu().numpy()
            conf_scores = boxes.conf.cpu().numpy()

            st.subheader("📊 Class Distribution")
            st.bar_chart(pd.Series(class_ids).value_counts())

            st.subheader("📈 Confidence Distribution")
            fig, ax = plt.subplots()
            ax.hist(conf_scores, bins=10)
            ax.set_xlabel("Confidence")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
        else:
            st.info("No detections available.")

# -------------------------------------------------
# DATASET & METRICS TAB
# -------------------------------------------------
with tab3:

    st.subheader("📂 Dataset Images")

    dataset_image_path = "datasets/images"

    if os.path.exists(dataset_image_path):

        image_files = [
            f for f in os.listdir(dataset_image_path)
            if f.lower().endswith((".jpg",".png",".jpeg"))
        ]

        if image_files:

            selected_dataset_image = st.selectbox(
                "Select Dataset Image",
                image_files
            )

            if selected_dataset_image:
                img_path = os.path.join(dataset_image_path, selected_dataset_image)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                st.image(img, use_column_width=True)

        else:
            st.warning("No images found in dataset.")
    else:
        st.warning("Dataset folder not found.")

    st.divider()

    st.subheader("🧪 Run Model Evaluation")

    dataset_yaml = "datasets/data.yaml"

    if os.path.exists(dataset_yaml):

        if st.button("Run Validation on Dataset"):

            model = get_model(model_option)

            with st.spinner("Evaluating Model..."):
                metrics = model.val(data=dataset_yaml)

                results_dict = metrics.results_dict

                st.subheader("📊 Evaluation Metrics")

                col1, col2, col3 = st.columns(3)
                col1.metric("mAP50", f"{results_dict.get('metrics/mAP50(B)',0):.3f}")
                col2.metric("mAP50-95", f"{results_dict.get('metrics/mAP50-95(B)',0):.3f}")
                col3.metric("Precision", f"{results_dict.get('metrics/precision(B)',0):.3f}")

                st.metric("Recall", f"{results_dict.get('metrics/recall(B)',0):.3f}")

                # Precision-Recall Curve
                st.subheader("📈 Precision-Recall Curve")
                if hasattr(metrics, "curves"):
                    pr_curve = metrics.curves.get("precision-recall")
                    if pr_curve is not None:
                        fig2, ax2 = plt.subplots()
                        ax2.plot(pr_curve[0], pr_curve[1])
                        ax2.set_xlabel("Recall")
                        ax2.set_ylabel("Precision")
                        st.pyplot(fig2)

                # Confusion Matrix
                st.subheader("🔢 Confusion Matrix")

                if hasattr(metrics, "confusion_matrix"):
                    cm = metrics.confusion_matrix.matrix

                    fig3, ax3 = plt.subplots(figsize=(8,6))
                    ax3.imshow(cm, cmap="Blues")
                    ax3.set_title("Confusion Matrix")
                    plt.colorbar(ax3.imshow(cm, cmap="Blues"))
                    st.pyplot(fig3)

    else:
        st.warning("datasets/data.yaml not found.")
