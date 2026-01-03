# python -m venv venv
# .\venv\Scripts\activate
# pip install ultralytics
# pip freeze > requirements.txt
# https://universe.roboflow.com/university-km5u7/traffic-sign-detection-yolov8-awuus/dataset/11

from ultralytics import YOLO

model = YOLO("yolov8n.pt")  

model.train(
    data ="traffic-sign-detection/data.yaml",
    epochs=2,
    imgsz=640,
    batch=16,
    name="traffic-sign-model",
    lr0=0.01,
    optimizer="SGD",
    weight_decay=0.0005,
    momentum=0.935,
    patience=50,
    workers=2,
    device="cpu",
    save = True,
    save_period=1,
    val = True,
    verbose = True,
)