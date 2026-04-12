from ultralytics import YOLO

def train():
    model = YOLO("yolov8s.pt")

    model.train(
        data="./yolo_dataset.yaml",
        imgsz=800,
        epochs=80,
        batch=16,
        device=0,
        workers=8,

        optimizer="AdamW",
        lr0=1e-3,
        weight_decay=5e-4,

        close_mosaic=10,
        patience=20,
        amp=True,

        project="runs/detect",
        name="food_detect_v8s",
        exist_ok=True
    )

if __name__ == "__main__":
    train()
