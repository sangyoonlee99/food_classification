# service/schemas.py

def build_inference_response(image_id: str, results: list) -> dict:
    """
    inference + ui_policy 결과를
    서비스 응답(JSON) 형태로 변환

    Args:
        image_id: 이미지 식별자
        results: core.inference + ui_policy 결과

    Returns:
        dict (API response)
    """
    return {
        "image_id": image_id,
        "detections": [
            {
                "bbox": r["bbox"],
                "ui_mode": r["ui_mode"],
                "top1": r["top1"],
                "candidates": r["candidates"]
            }
            for r in results
        ]
    }
