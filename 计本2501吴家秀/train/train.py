from ultralytics import YOLO
model=YOLO("yolov8n.pt")
results=model.train(data="D:/yolo/data.yaml",epochs=50,imgsz=640,batch=8)