# 🚀 Real-Time Object Detection using YOLOv8

## 📌 Project Overview
This project implements an end-to-end Object Detection system using YOLOv8.
The system can train, evaluate, and perform inference on images and videos.

---

## 🎯 Problem Statement
Build a custom object detector that:
- Trains on a labeled dataset
- Evaluates using standard detection metrics
- Performs inference on new images and videos
- Generates annotated outputs

---

## 🗂 Dataset
Classes:
- Ambulance
- Bus
- Car
- Motorcycle
- Truck
- Person

Dataset prepared in YOLO format:

images/train
images/val
labels/train
labels/val
data.yaml


---

## 🧠 Model Details
- Model: YOLOv8n
- Epochs: 100
- Image Size: 800
- Batch Size: 16

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| mAP@0.5 | 0.57 |
| mAP@0.5:0.95 | 0.40 |
| Precision | 0.66 |
| Recall | 0.53 |

---

## 📈 Observations
- Strong performance on Car detection.
- Moderate detection for Bus and Ambulance.
- Person detection can be improved.
- Some confusion between vehicle classes.

---

## 🎥 Inference Capabilities
- Image detection
- Video detection
- Annotated output saving
- Streamlit-based demo interface

---

## 🛠 How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py

Author
Ashish Mehara
