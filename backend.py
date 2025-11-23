from flask import Flask, request, jsonify
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

app = Flask(__name__)

model = models.resnet18(pretrained=True)
model.eval()

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

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files['file']
    img = Image.open(file.stream).convert('RGB')
    img_t = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_t)
        _, pred = torch.max(outputs, 1)
    label = "Dog" if pred.item() % 2 else "Cat"
    return jsonify({"result": label})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
