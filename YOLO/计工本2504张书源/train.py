from ultralytics import YOLO


def xunlianmoxing():
    print("开始训练")

    model = YOLO('yolov8n.pt')

    training_config = {
        'data': 'data/license_plate.yaml',
        'epochs': 50,
        'imgsz': 640,
        'batch': 16,
        'device': 'cpu',
        'workers': 2,
        'name': 'plate_detector',
    }

    results = model.train(**training_config)
    print(f"储存到: runs/detect/{training_config['name']}/weights/best.pt")
    return results

if __name__ == "__main__":
    xunlianmoxing()