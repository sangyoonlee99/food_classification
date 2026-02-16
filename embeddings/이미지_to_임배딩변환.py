from torchvision import transforms
from PIL import Image
import torch
import numpy as np

IMG_SIZE = 352

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])

def image_to_embedding(img_input, embedder, device):
    """
    img_input:
      - Path / str  → 파일에서 로드
      - np.ndarray  → crop된 이미지
    """

    # 1️⃣ 입력 타입 분기
    if isinstance(img_input, (str, bytes)) or hasattr(img_input, "__fspath__"):
        img = Image.open(img_input).convert("RGB")

    elif isinstance(img_input, np.ndarray):
        img = Image.fromarray(img_input[..., ::-1]).convert("RGB")
        # ↑ OpenCV(BGR) → PIL(RGB)

    else:
        raise TypeError(f"Unsupported input type: {type(img_input)}")

    # 2️⃣ Transform
    x = transform(img).unsqueeze(0).to(device)

    # 3️⃣ Embedding
    with torch.inference_mode():
        with torch.cuda.amp.autocast(False):
            emb = embedder(x)   # (1, D)

    return emb.squeeze(0).cpu()   # (D,)

