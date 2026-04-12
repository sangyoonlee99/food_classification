from ultralytics import YOLO
import torch

def main():
  
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    # 경로 설정
    PRETRAINED_MODEL = r"C:\Users\asia\PycharmProjects\food_classification\Yolo\near_hard\runs\detect\food_detect_v8s_nearhard_finetune\weights\best.pt"
    DATA_YAML = r"C:\Users\asia\PycharmProjects\food_classification\Yolo\yolo_robo_재학습\robo_data\robo_data.yaml"


    # 모델 로드
    model = YOLO(PRETRAINED_MODEL)

    # =========================
    # 재학습 (Fine-tuning)
    # =========================
    model.train(
        data=DATA_YAML,
        epochs=8,                 # 짧게
        imgsz=640,
        batch=16,
        lr0=5e-4,                 # 기존 대비 낮게
        lrf=0.01,
        optimizer="AdamW",        # 아담w적용 from 논문
        momentum=0.937,
        weight_decay=5e-4,
        cos_lr=True,
        close_mosaic=5,         
        patience=5,             
        device=device,
        verbose=True
    )

if __name__ == "__main__":
    main()
