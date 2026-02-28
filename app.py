import streamlit as st
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
import json
from ultralytics import YOLO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

st.set_page_config(page_title="YOLOv8 Pro Dashboard", layout="wide")
st.title("🚀 YOLOv8 Professional ML Dashboard")

# -------------------------
# MODEL LOADER (LAZY)
# -------------------------
@st.cache_resource
def load_model(path):
    return YOLO(path)

def get_model(name):
    if "models" not in st.session_state:
        st.session_state.models = {}
    if name not in st.session_state.models:
        st.session_state.models[name] = load_model(name)
    return st.session_state.models[name]

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.header("⚙ Settings")

model_option = st.sidebar.selectbox(
    "Select Model",
    ["yolov8n.pt", "yolov8s.pt", "best.pt"]
)

confidence = st.sidebar.slider("Confidence", 0.1, 1.0, 0.30)
iou = st.sidebar.slider("IoU", 0.1, 1.0, 0.45)
imgsz = st.sidebar.selectbox("Image Size", [320,480,640], index=2)

# -------------------------
# TABS
# -------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔍 Detection", "📊 Analytics", "🧪 Dataset & Metrics", "⚖ Model Compare"]
)

# ============================================================
# 🔍 DETECTION
# ============================================================
with tab1:

    uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if uploaded:
        model = get_model(model_option)

        img_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        image = cv2.imdecode(img_bytes,1)

        start = time.time()
        results = model.predict(source=image, conf=confidence, iou=iou, imgsz=imgsz)
        end = time.time()

        fps = 1/(end-start)
        boxes = results[0].boxes

        annotated = results[0].plot()
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        st.image(annotated, use_column_width=True)
        st.metric("FPS", f"{fps:.2f}")
        st.metric("Objects", len(boxes))

        # CSV Export
        if len(boxes) > 0:
            df = pd.DataFrame({
                "Class_ID": boxes.cls.cpu().numpy(),
                "Confidence": boxes.conf.cpu().numpy()
            })
            csv = df.to_csv(index=False).encode()
            st.download_button("📥 Download Detection CSV", csv, "detections.csv")

# ============================================================
# 📊 ANALYTICS
# ============================================================
with tab2:

    if uploaded:
        boxes = results[0].boxes

        if len(boxes) > 0:
            class_ids = boxes.cls.cpu().numpy()
            conf_scores = boxes.conf.cpu().numpy()

            st.subheader("Class Distribution")
            st.bar_chart(pd.Series(class_ids).value_counts())

            st.subheader("Confidence Distribution")
            fig, ax = plt.subplots()
            ax.hist(conf_scores, bins=10)
            st.pyplot(fig)

# ============================================================
# 🧪 DATASET & METRICS
# ============================================================
with tab3:

    dataset_yaml = "datasets/data.yaml"
    dataset_images = "datasets/images"
    dataset_labels = "datasets/labels"

    st.subheader("📂 Dataset Viewer")

    if os.path.exists(dataset_images):
        files = [f for f in os.listdir(dataset_images)
                 if f.lower().endswith((".jpg",".png",".jpeg"))]

        if files:
            selected = st.selectbox("Select Dataset Image", files)
            img_path = os.path.join(dataset_images, selected)
            img = cv2.imread(img_path)

            # Draw ground truth boxes
            label_path = os.path.join(dataset_labels,
                        selected.replace(".jpg",".txt")
                                .replace(".png",".txt")
                                .replace(".jpeg",".txt"))

            if os.path.exists(label_path):
                h, w, _ = img.shape
                with open(label_path) as f:
                    for line in f.readlines():
                        cls, x, y, bw, bh = map(float, line.split())
                        x1 = int((x - bw/2) * w)
                        y1 = int((y - bh/2) * h)
                        x2 = int((x + bw/2) * w)
                        y2 = int((y + bh/2) * h)
                        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img, use_column_width=True)

    st.divider()

    st.subheader("📊 Run Evaluation")

    if os.path.exists(dataset_yaml):
        if st.button("Run Validation"):
            model = get_model(model_option)
            metrics = model.val(data=dataset_yaml)
            results_dict = metrics.results_dict

            # Display Metrics
            st.metric("mAP50", results_dict.get("metrics/mAP50(B)",0))
            st.metric("mAP50-95", results_dict.get("metrics/mAP50-95(B)",0))
            st.metric("Precision", results_dict.get("metrics/precision(B)",0))
            st.metric("Recall", results_dict.get("metrics/recall(B)",0))

            # Save Accuracy Log
            log_data = {
                "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "model": model_option,
                "mAP50": results_dict.get("metrics/mAP50(B)",0),
                "precision": results_dict.get("metrics/precision(B)",0),
                "recall": results_dict.get("metrics/recall(B)",0)
            }

            df_log = pd.DataFrame([log_data])
            if os.path.exists("evaluation_logs.csv"):
                df_log.to_csv("evaluation_logs.csv", mode="a", header=False, index=False)
            else:
                df_log.to_csv("evaluation_logs.csv", index=False)

            # Accuracy History Graph
            history = pd.read_csv("evaluation_logs.csv")
            st.subheader("📈 Accuracy History")
            st.line_chart(history.set_index("date")[["mAP50"]])

            # Confusion Matrix with Class Names
            if hasattr(metrics,"confusion_matrix"):
                cm = metrics.confusion_matrix.matrix
                names = metrics.names

                fig, ax = plt.subplots(figsize=(8,6))
                im = ax.imshow(cm, cmap="Blues")
                ax.set_xticks(np.arange(len(names)))
                ax.set_yticks(np.arange(len(names)))
                ax.set_xticklabels(names.values(), rotation=90)
                ax.set_yticklabels(names.values())
                plt.colorbar(im)
                st.pyplot(fig)

            # PDF Export
            if st.button("📄 Export PDF Report"):
                doc = SimpleDocTemplate("evaluation_report.pdf", pagesize=A4)
                elements = []
                style = ParagraphStyle(name='Normal', fontSize=12)

                elements.append(Paragraph("YOLOv8 Evaluation Report", style))
                elements.append(Spacer(1,0.5*inch))

                for k,v in results_dict.items():
                    elements.append(Paragraph(f"{k}: {v}", style))
                    elements.append(Spacer(1,0.2*inch))

                doc.build(elements)

                with open("evaluation_report.pdf","rb") as f:
                    st.download_button(
                        "Download PDF",
                        f.read(),
                        "evaluation_report.pdf"
                    )

# ============================================================
# ⚖ MODEL COMPARE
# ============================================================
with tab4:

    model_a = st.selectbox("Model A", ["yolov8n.pt","best.pt"])
    model_b = st.selectbox("Model B", ["yolov8n.pt","best.pt"])

    if st.button("Compare Models"):
        m1 = get_model(model_a)
        m2 = get_model(model_b)

        res1 = m1.val(data="datasets/data.yaml").results_dict
        res2 = m2.val(data="datasets/data.yaml").results_dict

        df_compare = pd.DataFrame([res1,res2], index=[model_a,model_b])
        st.dataframe(df_compare)

# ============================================================
# MODEL VERSION TRACKING
# ============================================================
if os.path.exists("models_metadata.json"):
    with open("models_metadata.json") as f:
        metadata = json.load(f)

    st.subheader("📌 Model Version History")
    st.table(metadata)
