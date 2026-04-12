# Hard Case 기반 재학습 코드
#
# **실험 결과**
# Hard Case 재학습 전: Hard case ratio 0.429 (150/350)
# Hard Case 재학습 후: Hard case ratio 0.431 (151/350)

# 성능 향상 없음 → Near-Hard Case 전략으로 전환

from ultralytics import YOLO

def finetune():
    model = YOLO(Path("./weights/yolo_best.pt"))
    model.train(
        data=Path("./hardcase.yaml"),

        imgsz=960,
        epochs=25,
        batch=8,

        lr0=5e-4,
        optimizer="AdamW",
        weight_decay=5e-4,

        close_mosaic=0,
        warmup_epochs=5,
        patience=10,

        # augmentation 최소화 (hard case 안정)
        hsv_h=0.01,
        hsv_s=0.4,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,

        device=0,
        workers=8,

        project="runs/detect",
        name="food_detect_v8s_hard_finetune",
        exist_ok=True,

        resume=False
    )

if __name__ == "__main__":
    finetune()
