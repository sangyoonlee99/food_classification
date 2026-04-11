# food_ai/embeddings.py
import torch
import timm
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

IMG_SIZE = 352

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

class EfficientNetEmbedding(torch.nn.Module):
    def __init__(self, model_name: str, weight_path: str, device: str):
        super().__init__()
        self.device = device

        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
        )

        state = torch.load(weight_path, map_location=device)
        self.backbone.load_state_dict(state, strict=False)

        self.backbone.eval().to(device)
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.emb_dim = getattr(self.backbone, "num_features", None) or 0

    def forward(self, x):
        with torch.inference_mode():
            feat = self.backbone(x)
            feat = F.normalize(feat, dim=1)
        return feat


def image_to_embedding(img_input, embedder, device: str = "cpu"):
    """
    img_input:
      - Path/str -> 파일에서 로드
      - np.ndarray (OpenCV BGR) -> crop 이미지
    """
    if isinstance(img_input, (str, bytes)) or hasattr(img_input, "__fspath__"):
        img = Image.open(img_input).convert("RGB")
    elif isinstance(img_input, np.ndarray):
        if img_input.ndim != 3 or img_input.shape[2] != 3:
            raise ValueError(f"Invalid ndarray shape: {img_input.shape}")
        img = Image.fromarray(img_input[..., ::-1]).convert("RGB")
    else:
        raise TypeError(f"Unsupported input type: {type(img_input)}")

    x = transform(img).unsqueeze(0).to(device)

    with torch.inference_mode():
        emb = embedder(x)  # (1, D)

    return emb.squeeze(0)  # (D,)
