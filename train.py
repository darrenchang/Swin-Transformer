import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models.swin_transformer_v2 import SwinTransformerV2


def main():
    # ===================== 基本設定 =====================
    data_dir = f"{Path.home()}/datasets/fish/train"
    num_epochs = 100
    batch_size = 32
    lr = 1e-4
    num_workers = os.cpu_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===================== 資料預處理 =====================
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]
    )

    train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    num_classes = len(train_dataset.classes)
    print(train_dataset.classes)
    with open(f"{Path.home()}/models/model.json", "w") as fd:
        thing_class = {"class_names": train_dataset.classes}
        fd.write(json.dumps(thing_class))

    # ===================== 建立模型 =====================
    model = SwinTransformerV2(
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0,
        drop_path_rate=0.1,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        pretrained_window_sizes=[0, 0, 0, 0],
        device=device,
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # ===================== 訓練迴圈 =====================
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Train one epoch - pass through the entire dataset once
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, top2_indices = torch.topk(outputs, k=2, dim=1)
            # 計算 top-2 命中數
            correct += (top2_indices == labels.view(-1, 1)).any(dim=1).sum().item()

            total += labels.size(0)
            # correct += top2_indices.eq(labels).sum().item()
        print(
            f"Epoch {epoch+1}: TrainLoss={running_loss/len(train_loader):.4f}, TrainAccuracy={100.*correct/total:.2f}%"
        )

        # TODO: Validate the model and print the score

    # ===================== 儲存模型 =====================
    torch.save(model.state_dict(), f"{Path.home()}/models/model.pth")


if __name__ == "__main__":
    main()
