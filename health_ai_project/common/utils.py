def load_classes_txt(path: str) -> dict[int, str]:
    """
    classes.txt → {class_id: food_name}
    """
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            mapping[idx] = line.strip()
    return mapping
