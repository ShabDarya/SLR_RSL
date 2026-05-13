import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import json

SEQUENCE_LENGTH = 48
MAX_HANDS = 2
NUM_LANDMARKS = 21 #количество точек для каждой руки
NUM_FEATURES_PER_LANDMARK_HAND = 3 #количество координат для каждой точки
NUM_FEATURES = MAX_HANDS * NUM_LANDMARKS * NUM_FEATURES_PER_LANDMARK_HAND  #21 точка по 3 координаты

NUM_FEATURES_PER_LANDMARK = 2
LIPS_IDX = [61, 37, 0, 267, 291, 405, 17, 181] #точки губ
NUM_FEATURES_FACE = len(LIPS_IDX)

POSE_IDX = [16, 14, 12, 11, 13, 15, 0] #точки рук + нос
NUM_FEATURES_POSE = len(POSE_IDX)

NUM_FEATURES_ALL = NUM_FEATURES + NUM_FEATURES_FACE * NUM_FEATURES_PER_LANDMARK + NUM_FEATURES_POSE * NUM_FEATURES_PER_LANDMARK


class LandmarkEmbedding(nn.Module):
    def __init__(self, in_features, d_model=256, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(in_features, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)

        attn_out, _ = self.attn(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm2(x + self.ffn(x))

        x = x.mean(dim=1)   
        return x

class GestureModel(nn.Module):
    def __init__(self, num_classes=1001, d_model=256, dropout=0.25):
        super().__init__()

        self.d_model = d_model

        self.hand_emb = LandmarkEmbedding(in_features=126, d_model=d_model, dropout=dropout)
        self.lips_emb  = LandmarkEmbedding(in_features=16,  d_model=d_model, dropout=dropout)
        self.pose_emb  = LandmarkEmbedding(in_features=14,  d_model=d_model, dropout=dropout)

        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.temporal_pos = nn.Parameter(torch.randn(1, 300, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model*3,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.pooling = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        with open("idx2label.json", "r", encoding="utf-8") as f:
            self.idx_to_text = json.load(f)

    def forward(self, x):
        hands = x[:, :, :126]     
        lips  = x[:, :, 126:142]   
        pose  = x[:, :, 142:]     
        h = self.hand_emb(hands)
        l = self.lips_emb(lips)
        p = self.pose_emb(pose)
        fused = torch.cat([h, l, p], dim=1)   
        fused = self.fusion(fused)          
        x = fused.unsqueeze(1) + self.temporal_pos[:, :1, :]  
        x = self.transformer(x)
        x = x.squeeze(1)                      
        logits = self.classifier(x)

        return logits
    
    def idx2text(self, x):
        probabilities = torch.softmax(x, dim=-1)
        confidence, predicted_class = torch.max(probabilities, dim=-1)
        predicted_text = self.idx_to_text[str(predicted_class.item())]
        return predicted_text