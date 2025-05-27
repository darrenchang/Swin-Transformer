import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from tqdm import tqdm
from collections import defaultdict

def main():
    # ===================== 基本設定 =====================
    data_dir = r"D:/桌面/AI vscode/cover_tap/datasets"
    num_epochs = 10
    batch_size = 32
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===================== 資料預處理 =====================
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    num_classes = len(train_dataset.classes)
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

    # ===================== 建立模型 =====================
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # ===================== 訓練迴圈 =====================
    num_epochs = 30  # 訓練次數改為30次
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

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
            #correct += top2_indices.eq(labels).sum().item()

        print(f"Epoch {epoch+1}: Loss={running_loss/len(train_loader):.4f}, Accuracy={100.*correct/total:.2f}%")

    # ===================== 儲存模型 =====================
    torch.save(model.state_dict(), "swin_transformer_tiny3.pth")

if __name__ == '__main__':
    main()
