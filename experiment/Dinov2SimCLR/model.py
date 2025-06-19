import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

class DINOv2SimCLR(nn.Module):
    def __init__(self, feature_dim=256, model_name="facebook/dinov2-base"):
        super(DINOv2SimCLR, self).__init__()

        # Load DINOv2 model and processor
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)

        # Freeze the backbone if desired (optional)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Figure out output embedding dimension
        hidden_dim = self.backbone.config.hidden_size  # e.g., 768 for dinov2-base

        # Projection head - MLP
        self.g = nn.Sequential(
            nn.Linear(hidden_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim, bias=True)
        )

    def forward(self, x):
        # print("Backbone started")
        outputs = self.backbone(pixel_values=x)
        last_hidden_state = outputs.last_hidden_state  # [B, Patches+1, C]
        cls_token = last_hidden_state[:, 0]  # [B, C] - use CLS token as image representation

        feature = cls_token  # encoder output
        # print("Backbone finished")
        out = self.g(feature)  # projection head

        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
