import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os
import subprocess
import matplotlib.pyplot as plt
from collections import Counter
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

st.set_page_config(page_title="AI Vision Dashboard", layout="wide")

# -------------------- DARK CSS POLISH --------------------
st.markdown("""
<style>
.block-container {padding-top: 1rem;}
.stMetric {background-color: #1C1F26; padding: 15px; border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

# -------------------- MODEL CACHE --------------------
@st.cache_resource
def load_model(path):
    return YOLO(path)

# -------------------- SIDEBAR --------------------
st.sidebar.title("⚙ Control Panel")

task = st.sidebar.radio("Mode", ["Image", "Video", "Webcam"])

confidence = st.sidebar.slider("Confidence", 0.1, 1.0, 0.5)
iou = st.sidebar.slider("IOU", 0.1, 1.0, 0.45)

# Auto detect all .pt models
model_files = [f for f in os.listdir() if f.endswith(".pt")]

custom_model = st.sidebar.file_uploader("Upload Custom Model (.pt)", type=["pt"])

if custom_model:
    temp_model = tempfile.NamedTemporaryFile(delete=False)
    temp_model.write(custom_model.read())
    model_path = temp_model.name
elif model_files:
    model_path = st.sidebar.selectbox("Select Model", model_files)
else:
    st.error("No .pt model found.")
    st.stop()

model = load_model(model_path)

class_names = list(model.names.values())
selected_classes = st.sidebar.multiselect(
    "Filter Classes", class_names, default=class_names
)

# -------------------- TABS --------------------
tab1, tab2 = st.tabs(["🚀 Detection", "📊 Evaluation"])

# ================= DETECTION TAB =================
with tab1:
    st.title("🚀 Advanced YOLOv8 Object Detection")

    # -------- IMAGE MODE --------
    if task == "Image":
        uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

        if uploaded_file:
            image = Image.open(uploaded_file)
            img_array = np.array(image)

            col1, col2 = st.columns(2)
            col1.image(image, caption="Original", use_column_width=True)

            with st.spinner("Detecting..."):
                results = model(img_array, conf=confidence, iou=iou)

            filtered_boxes = [
                box for box in results[0].boxes
                if model.names[int(box.cls)] in selected_classes
            ]

            results[0].boxes = filtered_boxes
            result_img = results[0].plot()

            col2.image(result_img, caption="Detected", use_column_width=True)

            # ---- ANALYTICS ----
            counts = Counter(model.names[int(b.cls)] for b in filtered_boxes)
            conf_list = [float(b.conf) for b in filtered_boxes]

            st.divider()
            m1, m2 = st.columns(2)
            m1.metric("Total Objects", sum(counts.values()))
            if conf_list:
                m2.metric("Avg Confidence", round(np.mean(conf_list), 2))

            st.subheader("Class Counts")
            st.table(counts)

            # Histogram
            if conf_list:
                fig, ax = plt.subplots()
                ax.hist(conf_list, bins=10)
                ax.set_title("Confidence Distribution")
                st.pyplot(fig)

            # Bar Graph
            if counts:
                fig2, ax2 = plt.subplots()
                ax2.bar(counts.keys(), counts.values())
                plt.xticks(rotation=45)
                ax2.set_title("Per-Class Count")
                st.pyplot(fig2)

            # Save + Download
            cv2.imwrite("detected.jpg", cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
            with open("detected.jpg","rb") as f:
                st.download_button("⬇ Download Result", f, "detected.jpg")

    # -------- VIDEO MODE --------
    elif task == "Video":
        uploaded_video = st.file_uploader("Upload Video", type=["mp4"])

        if uploaded_video:
            temp_video = tempfile.NamedTemporaryFile(delete=False)
            temp_video.write(uploaded_video.read())

            cap = cv2.VideoCapture(temp_video.name)

            width = int(cap.get(3))
            height = int(cap.get(4))
            fps = cap.get(cv2.CAP_PROP_FPS)

            out = cv2.VideoWriter("output.mp4",
                                  cv2.VideoWriter_fourcc(*'mp4v'),
                                  fps, (width,height))

            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress = st.progress(0)
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame, conf=confidence, iou=iou)
                out.write(results[0].plot())

                frame_count += 1
                progress.progress(frame_count/total)

            cap.release()
            out.release()

            # Compress
            subprocess.run([
                "ffmpeg","-i","output.mp4",
                "-vcodec","libx264","-crf","28",
                "-preset","fast","-movflags","+faststart",
                "compressed.mp4"
            ])

            st.success("Video Processed")
            st.video("compressed.mp4")

            with open("compressed.mp4","rb") as f:
                st.download_button("⬇ Download Video", f, "detected_video.mp4")

    # -------- WEBCAM MODE --------
    elif task == "Webcam":

        class VideoProcessor(VideoTransformerBase):
            def transform(self, frame):
                img = frame.to_ndarray(format="bgr24")
                results = model(img, conf=confidence, iou=iou)
                img = results[0].plot()
                return av.VideoFrame.from_ndarray(img, format="bgr24")

        st.subheader("📷 Live Webcam Detection")
        webrtc_streamer(
            key="webcam",
            video_processor_factory=VideoProcessor
        )

# ================= EVALUATION TAB =================
with tab2:
    st.title("📊 Model Evaluation")

    st.metric("mAP50", 0.57)
    st.metric("mAP50-95", 0.40)
    st.metric("Precision", 0.66)
    st.metric("Recall", 0.53)

    st.info("Replace with real evaluation results if available.")        return None
    try:
        cloudinary.config(
            cloud_name=st.secrets["cloud_name"],
            api_key=st.secrets["api_key"],
            api_secret=st.secrets["api_secret"]
        )
        response = cloudinary.uploader.upload(file_path, resource_type="auto")
        return response["secure_url"]
    except:
        return None

# -------------------- TABS --------------------
tab1, tab2 = st.tabs(["🚀 Detection", "📊 Evaluation"])

# ================= DETECTION =================
with tab1:
    st.title("🚀 AI Vision Pro - YOLOv8")

    # -------- IMAGE MODE --------
    if mode == "Image":
        uploaded = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])
        if uploaded:
            image = Image.open(uploaded)
            img_np = np.array(image)

            col1, col2 = st.columns(2)
            col1.image(image, caption="Original", use_column_width=True)

            with st.spinner("Detecting..."):
                results = model(img_np, conf=confidence, iou=iou)

            filtered_boxes = [
                box for box in results[0].boxes
                if model.names[int(box.cls)] in selected_classes
            ]
            results[0].boxes = filtered_boxes
            result_img = results[0].plot()

            col2.image(result_img, caption="Detected", use_column_width=True)

            # ---- Analytics ----
            counts = Counter(model.names[int(b.cls)] for b in filtered_boxes)
            conf_list = [float(b.conf) for b in filtered_boxes]

            st.divider()
            m1, m2 = st.columns(2)
            m1.metric("Total Objects", sum(counts.values()))
            if conf_list:
                m2.metric("Avg Confidence", round(np.mean(conf_list), 2))

            st.subheader("Class Counts")
            st.table(counts)

            # Histogram
            if conf_list:
                fig, ax = plt.subplots()
                ax.hist(conf_list, bins=10)
                ax.set_title("Confidence Distribution")
                st.pyplot(fig)

            # Bar Graph
            if counts:
                fig2, ax2 = plt.subplots()
                ax2.bar(counts.keys(), counts.values())
                plt.xticks(rotation=45)
                ax2.set_title("Per-Class Count")
                st.pyplot(fig2)

            # Save result
            cv2.imwrite("detected.jpg",
                        cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))

            with open("detected.jpg","rb") as f:
                st.download_button("⬇ Download Result", f, "detected.jpg")

            # Cloud Upload
            if st.button("☁ Upload to Cloud"):
                url = upload_to_cloud("detected.jpg")
                if url:
                    st.success("Uploaded Successfully")
                    st.write(url)
                else:
                    st.warning("Cloud not configured.")

    # -------- VIDEO MODE --------
    elif mode == "Video":
        uploaded_vid = st.file_uploader("Upload Video", type=["mp4"])
        if uploaded_vid:
            temp_vid = tempfile.NamedTemporaryFile(delete=False)
            temp_vid.write(uploaded_vid.read())

            cap = cv2.VideoCapture(temp_vid.name)
            width = int(cap.get(3))
            height = int(cap.get(4))
            fps = cap.get(cv2.CAP_PROP_FPS)

            out = cv2.VideoWriter("output.mp4",
                                  cv2.VideoWriter_fourcc(*'mp4v'),
                                  fps, (width,height))

            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress = st.progress(0)
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame, conf=confidence, iou=iou)
                out.write(results[0].plot())
                frame_count += 1
                progress.progress(frame_count/total)

            cap.release()
            out.release()

            # Compress video
            subprocess.run([
                "ffmpeg","-i","output.mp4",
                "-vcodec","libx264","-crf","28",
                "-preset","fast","-movflags","+faststart",
                "compressed.mp4"
            ])

            st.success("Video Processed")
            st.video("compressed.mp4")

            with open("compressed.mp4","rb") as f:
                st.download_button("⬇ Download Video", f, "detected_video.mp4")

            if st.button("☁ Upload Video to Cloud"):
                url = upload_to_cloud("compressed.mp4")
                if url:
                    st.success("Uploaded Successfully")
                    st.write(url)

    # -------- WEBCAM MODE --------
    elif mode == "Webcam":
        class VideoProcessor(VideoTransformerBase):
            def transform(self, frame):
                img = frame.to_ndarray(format="bgr24")
                results = model(img, conf=confidence, iou=iou)
                img = results[0].plot()
                return av.VideoFrame.from_ndarray(img, format="bgr24")

        st.subheader("📷 Live Webcam Detection")
        webrtc_streamer(
            key="webcam",
            video_processor_factory=VideoProcessor
        )

# ================= EVALUATION =================
with tab2:
    st.title("📊 Model Evaluation")
    st.metric("mAP50", 0.57)
    st.metric("mAP50-95", 0.40)
    st.metric("Precision", 0.66)
    st.metric("Recall", 0.53)
    st.info("Replace with real evaluation metrics if available.")
            col1, col2 = st.columns(2)
            col1.image(image, caption="Original")

            results = model(img_np, conf=confidence, iou=iou)

            filtered_boxes = [
                box for box in results[0].boxes
                if model.names[int(box.cls)] in selected_classes
            ]

            results[0].boxes = filtered_boxes
            result_img = results[0].plot()

            col2.image(result_img, caption="Detected")

            counts = Counter(model.names[int(b.cls)] for b in filtered_boxes)
            conf_list = [float(b.conf) for b in filtered_boxes]

            st.metric("Total Objects", sum(counts.values()))
            st.table(counts)

            if len(conf_list) > 0:
                fig, ax = plt.subplots()
                ax.hist(conf_list, bins=10)
                ax.set_title("Confidence Histogram")
                st.pyplot(fig)

            if len(counts) > 0:
                fig2, ax2 = plt.subplots()
                ax2.bar(counts.keys(), counts.values())
                ax2.set_title("Per-Class Count")
                plt.xticks(rotation=45)
                st.pyplot(fig2)

            cv2.imwrite(
                "detected.jpg",
                cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
            )

            with open("detected.jpg", "rb") as f:
                st.download_button(
                    "⬇ Download Result",
                    f,
                    file_name="detected.jpg"
                )

    elif mode == "Video":

        uploaded_vid = st.file_uploader("Upload Video", type=["mp4"])

        if uploaded_vid is not None:

            temp_vid = tempfile.NamedTemporaryFile(delete=False)
            temp_vid.write(uploaded_vid.read())

            cap = cv2.VideoCapture(temp_vid.name)

            width = int(cap.get(3))
            height = int(cap.get(4))
            fps = cap.get(cv2.CAP_PROP_FPS)

            out = cv2.VideoWriter(
                "output.mp4",
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height)
            )

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress = st.progress(0)
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame, conf=confidence, iou=iou)
                out.write(results[0].plot())

                frame_count += 1
                progress.progress(frame_count / total_frames)

            cap.release()
            out.release()

            subprocess.run(
                [
                    "ffmpeg",
                    "-i", "output.mp4",
                    "-vcodec", "libx264",
                    "-crf", "28",
                    "-preset", "fast",
                    "-movflags", "+faststart",
                    "compressed.mp4"
                ]
            )

            st.video("compressed.mp4")

            with open("compressed.mp4", "rb") as f:
                st.download_button(
                    "⬇ Download Video",
                    f,
                    file_name="detected_video.mp4"
                )


# ================= EVALUATION =================
with tab2:

    st.title("📊 Evaluation")

    uploaded_images = st.file_uploader(
        "Upload Multiple Images for Evaluation",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_images:

        y_true = []
        y_pred = []
        y_scores = []

        for file in uploaded_images:
            img = Image.open(file)
            img_np = np.array(img)

            results = model(img_np, conf=confidence, iou=iou)

            for box in results[0].boxes:
                cls = int(box.cls)
                conf_score = float(box.conf)

                y_true.append(cls)
                y_pred.append(cls)
                y_scores.append(conf_score)

        if len(y_true) > 0:

            cm = confusion_matrix(y_true, y_pred)

            fig_cm, ax_cm = plt.subplots()
            ax_cm.imshow(cm)
            ax_cm.set_title("Confusion Matrix")
            st.pyplot(fig_cm)

            precision, recall, _ = precision_recall_curve(
                np.array(y_true) > 0,
                y_scores
            )

            fig_pr, ax_pr = plt.subplots()
            ax_pr.plot(recall, precision)
            ax_pr.set_title("Precision-Recall Curve")
            st.pyplot(fig_pr)

            map_score = float(np.mean(precision))
            st.metric("Approx mAP", round(map_score, 3))

            if st.button("Generate PDF Report"):

                pdf_file = "evaluation_report.pdf"
                doc = SimpleDocTemplate(pdf_file)
                elements = []
                styles = getSampleStyleSheet()

                elements.append(
                    Paragraph("YOLOv8 Evaluation Report", styles["Title"])
                )
                elements.append(Spacer(1, 0.3 * inch))
                elements.append(
                    Paragraph(f"Approx mAP: {round(map_score,3)}",
                              styles["Normal"])
                )

                fig_pr.savefig("pr_curve.png")

                elements.append(
                    RLImage("pr_curve.png", width=4 * inch, height=3 * inch)
                )

                doc.build(elements)

                with open(pdf_file, "rb") as f:
                    st.download_button(
                        "Download Evaluation PDF",
                        f,
                        file_name="YOLO_Evaluation_Report.pdf"
)
