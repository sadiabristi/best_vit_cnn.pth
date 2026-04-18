import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import timm

# ================= CONFIG =================
IMAGE_SIZE = 224
CLASSES = ["Healthy", "Mosaic", "RedRot", "Rust", "Yellow"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= MODEL =================
class CNNBranch(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        base = models.resnet50(pretrained=pretrained)
        self.features = nn.Sequential(*list(base.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)


class ViTBranch(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        self.vit.reset_classifier(0)

    def forward(self, x):
        return self.vit(x)


class HybridViTCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.vit = ViTBranch()
        self.cnn = CNNBranch()

        self.vit_proj = nn.Sequential(nn.Linear(768, 512), nn.LayerNorm(512))
        self.cnn_proj = nn.Sequential(nn.Linear(2048, 512), nn.LayerNorm(512))

        self.head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        v = self.vit(x)
        c = self.cnn(x)

        v = F.gelu(self.vit_proj(v))
        c = F.gelu(self.cnn_proj(c))

        fused = torch.cat([v, c], dim=1)
        return self.head(fused)


# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    model = HybridViTCNN(len(CLASSES))
    model.load_state_dict(torch.load("best_vit_cnn.pth", map_location=DEVICE))
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
st.title("🌱 Sugarcane Disease Detection (ViT + CNN)")

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
    st.write("Confidence:")
    for i, cls in enumerate(CLASSES):
        st.write(f"{cls}: {probs[0][i]*100:.2f}%")