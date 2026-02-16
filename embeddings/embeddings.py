# embeddings.py
import torch
import timm
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

# =========================
# Config
# =========================
IMG_SIZE = 352

# ⚠️ 반드시 학습 때와 동일해야 함
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])

# =========================
# EfficientNet → Embedding
# =========================
class EfficientNetEmbedding(torch.nn.Module):
    def __init__(self, model_name, weight_path, device):
        super().__init__()
        self.device = device

        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0   # 🔥 classifier 제거
        )

        # self.backbone.load_state_dict(
        #     torch.load(weight_path, map_location=device),
        #     strict=False
        # )
        self.backbone.load_state_dict(
            torch.load(weight_path, map_location=device, weights_only=True),
            strict=False
        )

        self.backbone.eval()
        self.backbone.to(device)

        # 🔒 파라미터 고정
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.emb_dim = self.backbone.num_features

    def forward(self, x):
        with torch.inference_mode():
            with torch.cuda.amp.autocast(False):
                feat = self.backbone(x)          # (B, D)
                feat = F.normalize(feat, dim=1)  # L2 normalize
        return feat

# =========================
# Image → Embedding
# =========================
def image_to_embedding(img_input, embedder, device):
    """
    img_input:
      - Path/str -> 파일에서 로드
      - np.ndarray (OpenCV BGR) -> crop 이미지
    """
    if isinstance(img_input, (str, bytes)) or hasattr(img_input, "__fspath__"):
        img = Image.open(img_input).convert("RGB")

    elif isinstance(img_input, np.ndarray):
        # OpenCV(BGR) -> PIL(RGB)
        if img_input.ndim != 3 or img_input.shape[2] != 3:
            raise ValueError(f"Invalid ndarray shape: {img_input.shape}")
        img = Image.fromarray(img_input[..., ::-1]).convert("RGB")

    else:
        raise TypeError(f"Unsupported input type: {type(img_input)}")

    x = transform(img).unsqueeze(0).to(device)

    with torch.inference_mode():
        with torch.cuda.amp.autocast(False):
            emb = embedder(x)   # (1, D)

    return emb.squeeze(0).cpu()