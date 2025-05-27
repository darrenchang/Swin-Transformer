import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import timm
import os
import json

# ===================== 設定參數 =====================
image_folder = r"D:/桌面/AI vscode/cover_tap/datasets/val/OutCircle"
model_path = "swin_transformer_tiny3.pth"
train_dir = r"D:/桌面/AI vscode/cover_tap/datasets/train/OutCircle"
#metadata_path = r"D:/桌面/AI vscode/2025_05_27-04_35_13/metadata.json"
save_folder = r"D:/桌面/AI vscode/cover_tap/datasets/result/OutCircle"
save_json_path = os.path.join(save_folder, "results.json")

#  train 的上層資料夾才對（包含所有類別資料夾）
class_names = sorted(os.listdir(os.path.dirname(train_dir)))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(save_folder, exist_ok=True)

# ===================== 載入模型 =====================
assert os.path.exists(model_path), f"模型檔不存在：{model_path}"
checkpoint = torch.load(model_path)
model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=len(class_names))

# 移除 head 層（若存在）
state_dict = checkpoint
for k in list(state_dict.keys()):
    if 'head.fc' in k:
        print(f"刪除 {k} 權重")
        del state_dict[k]

model.load_state_dict(state_dict, strict=False)
model.eval().to(device)

# ===================== 載入 metadata.json =====================
#with open(metadata_path, "r") as f:
#    metadata = json.load(f)

#uuid = metadata.get("uuid", "unknown")
#snapshot_from = metadata.get("snapshot_from", "unknown")
#model_config = metadata.get("model_config", "unknown")

# ===================== 預處理 =====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===================== 開始預測 =====================
results = []

for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(image_folder, filename)
        img = Image.open(img_path).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1).squeeze(0)

            # 取得 Top2
            top2_probs, top2_indices = torch.topk(probabilities, k=2)
            top2_classes = [class_names[idx] for idx in top2_indices]

            predicted_class = top2_classes[0]

        # ========== 儲存文字於圖片 ==========
        draw = ImageDraw.Draw(img)

        # 根據圖片寬度決定字體大小與 padding（整體再小一點）
        font_size = max(10, img.width //  30)
        padding = max(2, img.width // 150)
        margin = max(3, img.width // 100)


        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        text = f"Predicted: {predicted_class}"
        #text = f"Predicted: {predicted_class}\nFrom: {snapshot_from}"
        lines = text.split("\n")
        line_heights = []
        line_widths = []
        total_height = 0

        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            h = bbox[3] - bbox[1]
            w = bbox[2] - bbox[0]
            line_heights.append(h)
            line_widths.append(w)
            total_height += h

        max_width = max(line_widths)

        # 計算文字框的左上角位置
        x = img.width - max_width - 2 * padding - margin
        y = img.height - total_height - 2 * padding - margin

        # 畫背景框（白底）
        draw.rectangle(
            [x - padding, y - padding, x + max_width + padding, y + total_height + padding],
            fill="white"
        )

        # 寫上每行文字
        current_y = y
        for i, line in enumerate(lines):
            draw.text((x, current_y), line, fill="black", font=font)
            current_y += line_heights[i]

        # 儲存圖片
        img.save(os.path.join(save_folder, filename))

        # ========== 儲存 JSON 結果 ==========
        result_entry = {
            "filename": filename,
            #"uuid": uuid,
            "predicted_class": predicted_class,
            "top2": [
                {"label": top2_classes[0], "score": round(top2_probs[0].item(), 6)},
                {"label": top2_classes[1], "score": round(top2_probs[1].item(), 6)},
            ]
        }
        results.append(result_entry)
        
        # 把印出結果放在這裡，確保每張圖都會印
        print(f"檔案: {filename}")
        print(f"Top2索引: {top2_indices}")
        print(f"Top2類別: {top2_classes}")
        print(f"預測結果: {predicted_class}")

# 儲存至 results.json
with open(save_json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
