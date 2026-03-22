"""CLIP+MLP 编辑质量评估模型训练

使用预提取的 CLIP 特征训练 MLP，用 Pairwise Ranking Loss。

使用方法：
  python training/train_clip_mlp.py
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

BASE_DIR = Path(__file__).parent
LABELS_PATH = BASE_DIR / "data" / "synthetic" / "labels.json"
FEATURES_PATH = BASE_DIR / "features" / "clip_features.npz"
MODEL_SAVE_DIR = BASE_DIR / "models"


# ── 模型定义 ──────────────────────────────────────

class EditQualityMLP(nn.Module):
    """编辑质量评估 MLP：输入编辑前后 CLIP 特征拼接，输出质量分"""

    def __init__(self, clip_dim: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(clip_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, original_feat, edited_feat):
        x = torch.cat([original_feat, edited_feat], dim=-1)
        return self.mlp(x).squeeze(-1)


# ── 数据集 ────────────────────────────────────────

class PreferencePairDataset(Dataset):
    """偏好对数据集：从三元组中构建 (better, worse) 对"""

    def __init__(self, pairs: list[dict], features: dict):
        self.preference_pairs = []

        # 按原始帧分组
        groups = {}
        for p in pairs:
            key = p["original"]
            if key not in groups:
                groups[key] = []
            groups[key].append(p)

        # 从同一原始帧的不同编辑中构建偏好对
        for orig, edits in groups.items():
            if orig not in features:
                continue
            for i in range(len(edits)):
                for j in range(i + 1, len(edits)):
                    a, b = edits[i], edits[j]
                    if a["edited"] not in features or b["edited"] not in features:
                        continue
                    if abs(a["score"] - b["score"]) < 0.1:
                        continue  # 差距太小跳过

                    if a["score"] > b["score"]:
                        better, worse = a, b
                    else:
                        better, worse = b, a

                    self.preference_pairs.append({
                        "original": orig,
                        "better_edited": better["edited"],
                        "worse_edited": worse["edited"],
                        "margin": abs(a["score"] - b["score"]),
                    })

        self.features = features
        random.shuffle(self.preference_pairs)
        print(f"  构建了 {len(self.preference_pairs)} 组偏好对")

    def __len__(self):
        return len(self.preference_pairs)

    def __getitem__(self, idx):
        pair = self.preference_pairs[idx]
        orig_feat = torch.tensor(self.features[pair["original"]])
        better_feat = torch.tensor(self.features[pair["better_edited"]])
        worse_feat = torch.tensor(self.features[pair["worse_edited"]])
        margin = pair["margin"]
        return orig_feat, better_feat, worse_feat, margin


# ── 训练 ──────────────────────────────────────────

def train():
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # 加载数据
    print("加载标签...")
    with open(LABELS_PATH, "r") as f:
        data = json.load(f)
    pairs = data["pairs"]
    print(f"  {len(pairs)} 组三元组")

    print("加载 CLIP 特征...")
    features = dict(np.load(str(FEATURES_PATH), allow_pickle=True))
    print(f"  {len(features)} 张图片特征")

    # 划分训练/验证
    random.seed(42)
    random.shuffle(pairs)
    split = int(len(pairs) * 0.8)
    train_pairs = pairs[:split]
    val_pairs = pairs[split:]

    print(f"\n构建训练集...")
    train_dataset = PreferencePairDataset(train_pairs, features)
    print(f"构建验证集...")
    val_dataset = PreferencePairDataset(val_pairs, features)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)

    # 模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EditQualityMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    print(f"\n设备: {device}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"训练集: {len(train_dataset)} 对, 验证集: {len(val_dataset)} 对")
    print(f"\n开始训练...")

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(30):
        # 训练
        model.train()
        total_loss = 0.0
        n_batches = 0

        for orig, better, worse, margin in train_loader:
            orig, better, worse = orig.to(device), better.to(device), worse.to(device)

            score_better = model(orig, better)
            score_worse = model(orig, worse)

            # Pairwise Ranking Loss with margin
            loss = -torch.log(torch.sigmoid(score_better - score_worse) + 1e-8).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for orig, better, worse, margin in val_loader:
                orig, better, worse = orig.to(device), better.to(device), worse.to(device)
                score_better = model(orig, better)
                score_worse = model(orig, worse)
                correct += (score_better > score_worse).sum().item()
                total += len(orig)

        val_acc = correct / max(total, 1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            # 保存最优模型
            save_path = MODEL_SAVE_DIR / "edit_quality_mlp_best.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_acc": val_acc,
                "train_loss": avg_loss,
            }, str(save_path))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1:2d}/30 | "
                f"Loss: {avg_loss:.4f} | "
                f"Val Acc: {val_acc:.3f} | "
                f"Best: {best_val_acc:.3f} (epoch {best_epoch})"
            )

    # 保存最终模型
    final_path = MODEL_SAVE_DIR / "edit_quality_mlp_final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": 30,
        "val_acc": val_acc,
    }, str(final_path))

    print(f"\n{'='*50}")
    print(f"训练完成！")
    print(f"  最佳验证准确率: {best_val_acc:.3f} (epoch {best_epoch})")
    print(f"  最优模型: {MODEL_SAVE_DIR / 'edit_quality_mlp_best.pt'}")
    print(f"  最终模型: {final_path}")

    if best_val_acc >= 0.85:
        print(f"\n  ✓ 达到部署标准 (>85%)，可以替代 VLM 评估")
    elif best_val_acc >= 0.80:
        print(f"\n  △ 接近部署标准 (80-85%)，建议先替代 MCTS 模拟")
    else:
        print(f"\n  ✗ 未达标 (<80%)，建议增加数据或升级到 Phase 2b")


if __name__ == "__main__":
    train()
