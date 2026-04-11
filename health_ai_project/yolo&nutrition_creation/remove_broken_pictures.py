#깨진이미지에 대하여 이미지파일과 텍스트 파일 자동 삭제
from pathlib import Path
from PIL import Image

IMG_DIR = Path("D:/yolo_dataset/images")
LBL_DIR = Path("D:/yolo_dataset/labels")

removed = []

for split in ["train", "val"]:
    img_dir = IMG_DIR / split
    lbl_dir = LBL_DIR / split

    for img_path in img_dir.glob("*.jpg"):
        try:
            with Image.open(img_path) as img:
                img.verify()  # 🔥 깨진 이미지 검증
        except Exception:
            # 🔴 깨진 이미지 발견
            lbl_path = lbl_dir / (img_path.stem + ".txt")

            print(f"[REMOVE] 깨진 이미지: {img_path.name}")
            img_path.unlink(missing_ok=True)
            lbl_path.unlink(missing_ok=True)

            removed.append(img_path.name)

print(f"\n총 제거된 깨진 이미지 수: {len(removed)}")
