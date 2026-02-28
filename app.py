import streamlit as st
import os
import cv2
import time
from ultralytics import YOLO
from PIL import Image
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="YOLOv8 Pro Detection", layout="wide")
st.title("🚀 Real-Time Object Detection using YOLOv8")

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Settings")

model_option = st.sidebar.selectbox(
    "Select Model",
    ["COCO Model (yolov8n.pt)", "Custom Model (best.pt)"]
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model(option):
    if "COCO" in option:
        return YOLO("yolov8n.pt")
    else:
        return YOLO("best.pt")

model = load_model(model_option)

# ---------------- BRIGHTNESS LOGIC ----------------
def calculate_brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray.mean()

def auto_confidence(brightness):
    if brightness < 50:
        return 0.05
    elif brightness < 100:
        return 0.10
    else:
        return 0.25

# ---------------- METRICS ----------------
st.sidebar.subheader("📊 Evaluation Metrics")

if os.path.exists("metrics.txt"):
    with open("metrics.txt") as f:
        for line in f:
            st.sidebar.write(line.strip())

# ---------------- CONFUSION MATRIX ----------------
st.subheader("📈 Confusion Matrix")

if os.path.exists("confusion_matrix.png"):
    st.image("confusion_matrix.png")

# ---------------- DATASET SECTION ----------------
st.subheader("📂 Test on Dataset Files")

dataset_path = "datasets"

if os.path.exists(dataset_path):

    files = [
        f for f in os.listdir(dataset_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".mp4", ".avi"))
    ]

    if files:
        files = sorted(files)
        search = st.text_input("🔍 Search file")
        filtered = [f for f in files if search.lower() in f.lower()]
        selected_file = st.selectbox("Select File", filtered)

        if selected_file:
            file_path = os.path.join(dataset_path, selected_file)
            ext = selected_file.split(".")[-1].lower()

            # -------- IMAGE --------
            if ext in ["jpg", "jpeg", "png"]:

                img = cv2.imread(file_path)

                brightness = calculate_brightness(img)
                conf = auto_confidence(brightness)

                if brightness < 80:
                    img = cv2.convertScaleAbs(img, alpha=1.4, beta=30)

                start = time.time()
                results = model.predict(img, conf=conf)
                end = time.time()

                fps = 1 / (end - start)

                annotated = results[0].plot()

                # 🔥 COLOR FIX
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                st.image(annotated_rgb)
                st.success(f"FPS: {fps:.2f}")
                st.info(f"Auto Confidence: {conf} | Brightness: {brightness:.1f}")

            # -------- VIDEO --------
            elif ext in ["mp4", "avi"]:

                cap = cv2.VideoCapture(file_path)
                stframe = st.empty()

                frame_skip = 2
                frame_count = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    if frame_count % frame_skip != 0:
                        continue

                    frame = cv2.resize(frame, (640, 480))

                    brightness = calculate_brightness(frame)
                    conf = auto_confidence(brightness)

                    if brightness < 80:
                        frame = cv2.convertScaleAbs(frame, alpha=1.4, beta=30)

                    start = time.time()
                    results = model.predict(frame, conf=conf)
                    end = time.time()

                    fps = 1 / (end - start)

                    annotated = results[0].plot()

                    # 🔥 COLOR FIX
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                    stframe.image(annotated_rgb)
                    st.caption(f"FPS: {fps:.2f}")

                cap.release()

# ---------------- UPLOAD SECTION ----------------
st.subheader("📤 Upload Image or Video")

uploaded_file = st.file_uploader(
    "Upload Image or Video",
    type=["jpg", "jpeg", "png", "mp4", "avi"]
)

if uploaded_file:

    ext = uploaded_file.name.split(".")[-1].lower()

    # -------- IMAGE --------
    if ext in ["jpg", "jpeg", "png"]:

        image = Image.open(uploaded_file)
        img_array = np.array(image)

        brightness = calculate_brightness(img_array)
        conf = auto_confidence(brightness)

        if brightness < 80:
            img_array = cv2.convertScaleAbs(img_array, alpha=1.4, beta=30)

        start = time.time()
        results = model.predict(img_array, conf=conf)
        end = time.time()

        fps = 1 / (end - start)

        annotated = results[0].plot()

        # 🔥 COLOR FIX
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        st.image(annotated_rgb)
        st.success(f"FPS: {fps:.2f}")
        st.info(f"Auto Confidence: {conf} | Brightness: {brightness:.1f}")

    # -------- VIDEO --------
    elif ext in ["mp4", "avi"]:

        temp_path = "temp_video.mp4"

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        cap = cv2.VideoCapture(temp_path)
        stframe = st.empty()

        frame_skip = 2
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            frame = cv2.resize(frame, (640, 480))

            brightness = calculate_brightness(frame)
            conf = auto_confidence(brightness)

            if brightness < 80:
                frame = cv2.convertScaleAbs(frame, alpha=1.4, beta=30)

            start = time.time()
            results = model.predict(frame, conf=conf)
            end = time.time()

            fps = 1 / (end - start)

            annotated = results[0].plot()

            # 🔥 COLOR FIX
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            stframe.image(annotated_rgb)
            st.caption(f"FPS: {fps:.2f}")

        cap.release()

# ---------------- PROJECT REPORT ----------------
st.subheader("📄 Project Report")

with st.expander("Click to View Full Report"):
    st.markdown("""
    ### 🎯 Problem Statement
    End-to-end YOLOv8 Object Detection system for training, evaluation and deployment.

    ### 📊 Performance
    - mAP@0.5: 0.57
    - mAP@0.5:0.95: 0.40
    - Precision: 0.66
    - Recall: 0.53

    ### 🚀 Conclusion
    The model demonstrates reliable vehicle detection and real-time deployment
    with adaptive confidence handling for low-light scenarios.
    """)
