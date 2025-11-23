import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

import os
print("🧭 Current working directory:", os.getcwd())
print("📂 Files in current dir:", os.listdir())


# ==== 1. Định nghĩa mô hình CNN ====
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# ==== 2. Load model đã huấn luyện ====
model = SimpleCNN()
import os
model_path = os.path.join(os.path.dirname(__file__), "models", "model.pth")
print("🔍 Loading model from:", model_path)
model.load_state_dict(torch.load(model_path, map_location="cpu"))

model.eval()

# ==== 3. Tiền xử lý & dự đoán ====
from PIL import Image

def predict_image(img_path):
    image = Image.open(img_path).convert('RGB')  # 👈 mở ảnh đúng kiểu
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    return "Cat 🐱" if predicted.item() == 0 else "Dog 🐶"


# ==== 4. Giao diện Streamlit ====
st.title("🐾 ỨNG DỤNG NHẬN DẠNG MÈO & CHÓ 🐾")
st.write("Tải lên ảnh của mèo hoặc chó để mô hình dự đoán nhé!")

uploaded_file = st.file_uploader("📸 Chọn ảnh để nhận dạng", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Ảnh bạn đã tải lên", use_container_width=True)
    st.write("⏳ Đang dự đoán...")
    label = predict_image(image)
    st.success(f"✅ **Kết quả:** {label}")
