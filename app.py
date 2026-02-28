import streamlit as st
import os
import cv2
import time
import numpy as np
from ultralytics import YOLO
from PIL import Image

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

# ---------------- BRIGHTNESS + AUTO CONF ----------------
def calculate_brightness(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return gray.mean()

def auto_confidence(brightness):
    if brightness < 50:
        return 0.20
    elif brightness < 100:
        return 0.25
    else:
        return 0.30

# ---------------- SAFE DISPLAY ----------------
def display_bgr(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, use_container_width=True)

# ---------------- METRICS ----------------
st.sidebar.subheader("📊 Evaluation Metrics")

if os.path.exists("metrics.txt"):
    with open("metrics.txt") as f:
        for line in f:
            st.sidebar.write(line.strip())

# ---------------- CONFUSION MATRIX ----------------
st.subheader("📈 Confusion Matrix")

if os.path.exists("confusion_matrix.png"):
    st.image("confusion_matrix.png", use_container_width=True)

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

                img_bgr = cv2.imread(file_path)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                brightness = calculate_brightness(img_rgb)
                conf = auto_confidence(brightness)

                if brightness < 80:
                    img_rgb = cv2.convertScaleAbs(img_rgb, alpha=1.3, beta=25)

                start = time.time()
                results = model.predict(
                    img_rgb,
                    conf=conf,
                    iou=0.45,
                    max_det=50
                )
                end = time.time()

                fps = 1 / (end - start)
                annotated = results[0].plot()

                display_bgr(annotated)

                st.success(f"FPS: {fps:.2f}")
                st.info(f"Auto Confidence: {conf} | Brightness: {brightness:.1f}")

            # -------- VIDEO --------
            elif ext in ["mp4", "avi"]:

                cap = cv2.VideoCapture(file_path)
                stframe = st.empty()

                frame_skip = 2
                frame_count = 0

                while cap.isOpened():
                    ret, frame_bgr = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    if frame_count % frame_skip != 0:
                        continue

                    frame_bgr = cv2.resize(frame_bgr, (640, 480))
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                    brightness = calculate_brightness(frame_rgb)
                    conf = auto_confidence(brightness)

                    if brightness < 80:
                        frame_rgb = cv2.convertScaleAbs(frame_rgb, alpha=1.3, beta=25)

                    start = time.time()
                    results = model.predict(
                        frame_rgb,
                        conf=conf,
                        iou=0.45,
                        max_det=50
                    )
                    end = time.time()

                    fps = 1 / (end - start)
                    annotated = results[0].plot()
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                    stframe.image(annotated_rgb, use_container_width=True)
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
        img_rgb = np.array(image)

        brightness = calculate_brightness(img_rgb)
        conf = auto_confidence(brightness)

        if brightness < 80:
            img_rgb = cv2.convertScaleAbs(img_rgb, alpha=1.3, beta=25)

        start = time.time()
        results = model.predict(
            img_rgb,
            conf=conf,
            iou=0.45,
            max_det=50
        )
        end = time.time()

        fps = 1 / (end - start)
        annotated = results[0].plot()

        display_bgr(annotated)

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
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            frame_bgr = cv2.resize(frame_bgr, (640, 480))
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            brightness = calculate_brightness(frame_rgb)
            conf = auto_confidence(brightness)

            if brightness < 80:
                frame_rgb = cv2.convertScaleAbs(frame_rgb, alpha=1.3, beta=25)

            start = time.time()
            results = model.predict(
                frame_rgb,
                conf=conf,
                iou=0.45,
                max_det=50
            )
            end = time.time()

            fps = 1 / (end - start)
            annotated = results[0].plot()
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            stframe.image(annotated_rgb, use_container_width=True)
            st.caption(f"FPS: {fps:.2f}")

        cap.release()
