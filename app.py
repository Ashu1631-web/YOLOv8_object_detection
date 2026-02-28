import streamlit as st
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import os
from ultralytics import YOLO

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Ashish AI Vision", layout="wide")

st.markdown("""
<h1 style='text-align: center; color: #1F77B4;'>
🚀 Ashish AI Vision - YOLOv8 Detection Suite
</h1>
<hr>
""", unsafe_allow_html=True)

# -------------------------------------------------
# MODEL LOAD (CACHED)
# -------------------------------------------------
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# -------------------------------------------------
# SIDEBAR
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

# Class Filter
class_filter = st.sidebar.multiselect(
    "Filter Classes",
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
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔍 Detection", "📊 Analytics", "🧪 Validation", "📄 Documentation"]
)

# -------------------------------------------------
# DETECTION
# -------------------------------------------------
with tab1:

    mode = st.radio("Select Mode", ["Image", "Video"])

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
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            col1, col2 = st.columns([2, 1])

            with col1:
                st.image(annotated, use_column_width=True)

            with col2:
                st.metric("⚡ FPS", f"{fps:.2f}")
                st.metric("📦 Objects", len(results[0].boxes))

            _, buffer = cv2.imencode(
                ".jpg",
                cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            )

            st.download_button(
                "📥 Download Result Image",
                buffer.tobytes(),
                "result.jpg",
                "image/jpeg"
            )

# -------------------------------------------------
# ANALYTICS
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
            st.pyplot(fig)

        else:
            st.info("No detections available.")

# -------------------------------------------------
# VALIDATION
# -------------------------------------------------
with tab3:

    st.subheader("Model Validation")

    dataset_yaml = st.text_input(
        "Dataset YAML path (example: datasets/data.yaml)"
    )

    if st.button("Run Validation") and dataset_yaml:

        with st.spinner("Validating..."):

            metrics = model.val(data=dataset_yaml)

            st.write("### Evaluation Metrics")
            st.write(metrics.results_dict)

            if hasattr(metrics, "confusion_matrix"):
                cm = metrics.confusion_matrix.matrix
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                ax2.set_title("Confusion Matrix")
                st.pyplot(fig2)

# -------------------------------------------------
# DOCUMENTATION (README + REPORT)
# -------------------------------------------------
with tab4:

    st.subheader("📘 Project README")

    if os.path.exists("README.md"):
        with open("README.md", "r", encoding="utf-8") as f:
            readme_content = f.read()
            st.markdown(readme_content)
    else:
        st.warning("README.md not found in repository.")

    st.divider()

    st.subheader("📑 Project Report")

    if os.path.exists("project_report.pdf"):
        with open("project_report.pdf", "rb") as pdf_file:
            PDFbyte = pdf_file.read()

        st.download_button(
            label="📥 Download Project Report",
            data=PDFbyte,
            file_name="project_report.pdf",
            mime="application/pdf"
        )

        st.success("Report ready for download.")
    else:
        st.warning("project_report.pdf not found in repository.")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("""
<hr>
<center>
Built with ❤️ by Ashish | YOLOv8 Powered | Streamlit Cloud Ready
</center>
""", unsafe_allow_html=True)
