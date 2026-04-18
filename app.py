import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import timm
import gdown
import os

# ================= CONFIG =================
IMAGE_SIZE = 224
CLASSES = ["Healthy", "Mosaic", "RedRot", "Rust", "Yellow"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "best_vit_cnn.pth"
FILE_ID = "1pwUoLixTrTrees-VvRwevocXZlMKX6BG"

# ================= MODEL =================
class CNNBranch(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet50(weights=None)   # ❗ no download
        self.features = nn.Sequential(*list(base.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)


class ViTBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=False)
        self.vit.reset_classifier(0)

    def forward(self, x):
        return self.vit(x)


class HybridViTCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.vit = ViTBranch()
        self.cnn = CNNBranch()

        self.vit_proj = nn.Linear(768, 512)
        self.cnn_proj = nn.Linear(2048, 512)

        self.head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        v = self.vit(x)
        c = self.cnn(x)

        v = self.vit_proj(v)
        c = self.cnn_proj(c)

        fused = torch.cat([v, c], dim=1)
        return self.head(fused)


# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

    model = HybridViTCNN(len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


model = load_model()

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ================= UI =================
st.title("🌱 Sugarcane Disease Detection")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    st.success(f"Prediction: {CLASSES[pred]}")
