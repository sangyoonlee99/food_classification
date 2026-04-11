import os
import json
import csv

# -------------------------------------------------------
# 0. 사용자 설정
# -------------------------------------------------------
ROOT = r"D:\download\TL_라벨링\296.비전영역,_음식이미지_및_정보소개_텍스트_데이터\01-1.정식개방데이터\Training\02.라벨링데이터\TL"
#  ↑ TL1~TL16 이 들어있는 최상위 폴더

OUTPUT_LABEL = "./labels_tl"          # YOLO txt 저장 폴더 (Validation과 구분하려고 이름 다르게)
OUTPUT_NUTRITION = "./nutrition_tl.csv"
OUTPUT_META = "./meta_tl.csv"

os.makedirs(OUTPUT_LABEL, exist_ok=True)

# -------------------------------------------------------
# 1. 폴더명 → 대분류 / 중분류 매핑
# -------------------------------------------------------
cat1_map = {
    "A": "일상음식/한식",
    "B": "일반외식/패턴",
    "C": "개인 대체 메뉴",
    "D": "음료 및 차류"
}

cat2_map = {
    "01": "면",
    "02": "빵",
    "03": "죽",
    "04": "수프",
    "05": "샌드위치",
    "06": "기타",
    "07": "과일/과채음료",
    "08": "유제품",
    "09": "일차류",
    "10": "커피류",
    "11": "비빔/무침",
    "12": "일반외식",
    "13": "향토음식",
    "14": "외국음식"
}

# -------------------------------------------------------
# 2. class_name → class_id 자동 매핑
#    (필요하면 나중에 기존 class_id 매핑 불러오는 버전도 만들어줄 수 있음)
# -------------------------------------------------------
class_map = {}  # food_name -> class_id

def get_class_id(name):
    if name not in class_map:
        class_map[name] = len(class_map)
    return class_map[name]

# -------------------------------------------------------
# 3. YOLO bbox 변환 함수
# -------------------------------------------------------
def convert_bbox_to_yolo(bb, img_w, img_h):
    x = bb["x"]
    y = bb["y"]
    w = bb["width"]
    h = bb["height"]

    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h

    return x_center, y_center, w_norm, h_norm

# -------------------------------------------------------
# 4. nutrition / meta 저장 준비
# -------------------------------------------------------
nutrition_rows = []
meta_rows = []

# -------------------------------------------------------
# 5. 전체 폴더 탐색 (TL1~TL16 포함)
# -------------------------------------------------------
for dirpath, dirnames, filenames in os.walk(ROOT):
    for fname in filenames:
        if not fname.endswith(".json"):
            continue

        json_path = os.path.join(dirpath, fname)

        # ROOT 이후 상대 경로
        rel = os.path.relpath(json_path, ROOT)
        parts = rel.split(os.sep)
        # 예: ["TL1","A","13","A13001","30","정위","A_13_A13001_가자미구이_30_01.json"]

        # -----------------------------------
        # "A/B/C/D" 가 있는 index 자동 탐색
        # -----------------------------------
        cat1_idx = None
        for i, p in enumerate(parts):
            if p in cat1_map:  # "A","B","C","D"
                cat1_idx = i
                break

        if cat1_idx is None or cat1_idx + 5 >= len(parts):
            print(f"[WARN] 예상치 못한 경로 구조, 스킵: {rel}")
            continue

        # TL 폴더 (예: "TL1")
        tl_folder = parts[0] if cat1_idx > 0 else None

        cat1 = parts[cat1_idx]           # "A"/"B"/"C"/"D"
        cat2 = parts[cat1_idx + 1]       # "13", "14" ...
        keycode = parts[cat1_idx + 2]    # "A13001"
        distance = parts[cat1_idx + 3]   # "30" 같은 숫자 폴더 (촬영 거리/코드로 추정)
        angle_kor = parts[cat1_idx + 4]  # "정위" / "측면" 폴더

        cat1_name = cat1_map.get(cat1, "UNKNOWN")
        cat2_name = cat2_map.get(cat2, "UNKNOWN")

        # -----------------------------------
        # JSON 로드
        # -----------------------------------
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)["data"]

        img_info = data["image_info"]
        img_w = img_info["width"]
        img_h = img_info["height"]

        # 파일명 예: A_13_A13001_가자미구이_30_01.jpg
        #           0  1   2        3        4  5
        file_name = img_info["file_name"]
        try:
            food_name = file_name.split("_")[3]
        except Exception:
            food_name = file_name  # 혹시 포맷이 다르면 파일명 전체 사용

        # -----------------------------------
        # 2D annotation → YOLO bbox
        # -----------------------------------
        bb = data["2d_annotation"]
        x_c, y_c, w_n, h_n = convert_bbox_to_yolo(bb, img_w, img_h)

        class_id = get_class_id(food_name)

        # -----------------------------------
        # YOLO txt 저장
        # -----------------------------------
        txt_name = fname.replace(".json", ".txt")
        txt_path = os.path.join(OUTPUT_LABEL, txt_name)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")

        # -----------------------------------
        # nutrition 저장
        # -----------------------------------
        nut = data["nutrition"]
        nut["food_name"] = food_name
        nutrition_rows.append(nut)

        # -----------------------------------
        # meta 저장
        # -----------------------------------
        meta_row = {
            "json_path": json_path,
            "food_name": food_name,
            "class_id": class_id,
            "category1": cat1_name,
            "category2": cat2_name,
            "keycode": keycode,
            "distance": distance,      # "30" 폴더 정보
            "angle": angle_kor,        # "정위"/"측면"
            "yolo_label": txt_path
        }
        if tl_folder is not None:
            meta_row["TL"] = tl_folder  # TL1~TL16

        meta_rows.append(meta_row)

# -------------------------------------------------------
# 6. nutrition_tl.csv 저장 (UTF-8-SIG → 엑셀 한글 깨짐 방지)
# -------------------------------------------------------
if nutrition_rows:
    fieldnames = nutrition_rows[0].keys()
    with open(OUTPUT_NUTRITION, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(nutrition_rows)

print(f"[OK] Saved nutrition CSV → {OUTPUT_NUTRITION}")

# -------------------------------------------------------
# 7. meta_tl.csv 저장 (UTF-8-SIG)
# -------------------------------------------------------
if meta_rows:
    fieldnames = list(meta_rows[0].keys())
    with open(OUTPUT_META, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(meta_rows)

print(f"[OK] Saved meta CSV → {OUTPUT_META}")

# -------------------------------------------------------
# 8. class map 출력
# -------------------------------------------------------
print("\n===== CLASS MAP (음식명 → class_id) =====")
for name, idx in class_map.items():
    print(idx, ":", name)
