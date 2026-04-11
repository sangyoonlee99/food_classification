## prototype load / similarity ##

# core/prototype.py
import torch
import torch.nn.functional as F

class PrototypeMatcher:
    def __init__(self, proto_path, device):
        data = torch.load(proto_path, map_location="cpu")
        self.class_names = data["classes"]
        self.proto_matrix = torch.stack(
            [data["prototypes"][c] for c in self.class_names]
        ).to(device)

    def topk(self, emb, k=5):
        sims = F.cosine_similarity(
            emb.unsqueeze(0),
            self.proto_matrix,
            dim=1
        )
        scores, idxs = torch.topk(sims, k)

        return [
            {
                "class": self.class_names[i.item()],
                "similarity": round(float(s), 4)
            }
            for s, i in zip(scores, idxs)
        ]
