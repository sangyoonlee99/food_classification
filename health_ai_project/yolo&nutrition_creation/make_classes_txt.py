import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1) 엑셀 파일 경로 설정 (필요하면 경로만 수정하세요)
# --------------------------------------------------
TL_PATH = Path(r"D:\yolo_dataset\nutrition_tl.xlsx")  # train 메타데이터
VL_PATH = Path(r"D:\yolo_dataset\nutrition_vl.xlsx")  # val 메타데이터

# --------------------------------------------------
# 2) 저장할 classes.txt 경로
# --------------------------------------------------
CLASSES_TXT_PATH = Path(r"D:\yolo_dataset\classes.txt")

# --------------------------------------------------
# 3) 엑셀에서 food_name / class_id 컬럼 읽기
# --------------------------------------------------
print(f"[INFO] Train 메타데이터 읽는 중... {TL_PATH}")
tl = pd.read_excel(TL_PATH)

print(f"[INFO] Val 메타데이터 읽는 중... {VL_PATH}")
vl = pd.read_excel(VL_PATH)

# 필요한 컬럼만 사용
cols = ["food_name", "class_id"]
tl2 = tl[cols].dropna()
vl2 = vl[cols].dropna()

# --------------------------------------------------
# 4) train + val 결합 후, class_id 기준으로 중복 제거 & 정렬
# --------------------------------------------------
combined = (
    pd.concat([tl2, vl2])
    .drop_duplicates(subset=["class_id"])
    .sort_values("class_id")
)

# 클래스 개수 확인
num_classes = combined.shape[0]
print(f"[INFO] 고유 클래스 개수: {num_classes}개")  # 800개 예상

# --------------------------------------------------
# 5) class_id 순서대로 food_name만 뽑아서 리스트로 만들기
# --------------------------------------------------
class_names = combined["food_name"].astype(str).tolist()

# 첫 몇 개, 마지막 몇 개 샘플 출력
print("[DEBUG] 앞 5개:", class_names[:5])
print("[DEBUG] 뒤 5개:", class_names[-5:])

# --------------------------------------------------
# 6) classes.txt 파일로 저장
#     - 각 줄에 하나의 food_name
#     - 줄 순서 = class_id (0,1,2,...,799)
# --------------------------------------------------
CLASSES_TXT_PATH.parent.mkdir(parents=True, exist_ok=True)

with CLASSES_TXT_PATH.open("w", encoding="utf-8") as f:
    for name in class_names:
        f.write(name + "\n")

print(f"[OK] classes.txt 생성 완료 → {CLASSES_TXT_PATH}")
