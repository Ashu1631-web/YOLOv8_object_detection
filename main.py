from fastapi import FastAPI
from ultralytics import YOLO

app = FastAPI()
model = YOLO("best.pt")

@app.get("/")
def home():
    return {"status": "API running"}
