import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ==== 1. Mô hình CNN ====
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
model.load_state_dict(torch.load("models/model.pth", map_location="cpu"))
model.eval()

# ==== 3. Tiền xử lý ảnh ====
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ==== 4. Giao diện người dùng ====
st.title("🐶🐱 AI Image Classifier")
st.write("Phân loại ảnh chó và mèo bằng mô hình CNN")

uploaded_file = st.file_uploader("📸 Tải ảnh lên (JPG hoặc PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Ảnh tải lên", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        label = "Cat 🐱" if predicted.item() == 0 else "Dog 🐶"
        st.subheader(f"Kết quả dự đoán: {label}")
