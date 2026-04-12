# Near-Hard Case 기반 재학습 코드
#
# 실험 결과
# Near-Hard Case 재학습 전: 233건
# Near-Hard Case 재학습 후: 206건
# → 11.59% 감소 확인
# 단, 시각적 경계가 불명확하거나 객체 크기가 극히 작은 경우
# 데이터·라벨 한계로 개선 폭 제한적

from ultralytics import YOLO
from pathlib import Path

def finetune():
    #기존 학습된 모델 로드

    model = YOLO(Path("./weights/yolo_best.pt"))


    #Fine-tuning
    model.train(
        # near-hard + easy 혼합 train
        data= Path("./finetune_nearhard_easy/nearhardcase.yaml"),

        #작은 객체 대응
        imgsz=960,

        #near-hard 기준 학습량
        epochs=40,
        batch=8,

        #보정 학습용 learning rate
        lr0=1e-3,
        optimizer="AdamW",
        weight_decay=5e-4,

        #hard/near-hard mosaic 끔
        close_mosaic=0,

        # 안정적인 수렴
        warmup_epochs=5,
        patience=10,

        #augmentation 최소화 (분포 보존)
        hsv_h=0.01,
        hsv_s=0.4,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,

        # 시스템
        device=0,
        workers=8,

        # 결과 저장
        project="runs/detect",
        name="food_detect_v8s_nearhard_finetune",
        exist_ok=True,

        #optimizer state 새로 시작
        resume=False
    )

if __name__ == "__main__":
    finetune()
