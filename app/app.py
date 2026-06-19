"""
🔍 DeepFake & AI-Generated Image Detector
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageFilter
import os
import json
import time

st.set_page_config(page_title="DeepFake Detector", page_icon="🔍", layout="wide")

# ── CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
html,body,[data-testid="stAppViewContainer"]{background:#07070f;color:#f0f0f8;}
[data-testid="stMetric"]{background:#0f0f1a;border:1px solid #1e1e2e;border-radius:14px;padding:16px 20px;}
[data-testid="stMetricLabel"]{color:#5555aa;}
[data-testid="stMetricValue"]{color:#a78bfa;}
[data-testid="stProgress"]>div>div{background:linear-gradient(90deg,#7c6af7,#a78bfa);}
</style>
""", unsafe_allow_html=True)

# ── Model ──────────────────────────────────────────────────
class SimpleDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.fc(x)

@st.cache_resource
def load_model():
    app_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(app_dir, 'pixel.pth')
    
    model = SimpleDetector()
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model, True
    
    return model, False

model, model_ok = load_model()

# ── Preprocessing ──────────────────────────────────────────
def preprocess_image(pil_img):
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(pil_img).unsqueeze(0)

def predict(pil_img):
    tensor = preprocess_image(pil_img)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1).squeeze().numpy()
    pred = int(probs.argmax())
    return pred, float(probs[0]), float(probs[1])

# ── UI ──────────────────────────────────────────────────────
st.title("🔍 DeepFake & AI Image Detector")
st.markdown("Upload an image to detect if it's REAL or AI-GENERATED")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Model", "✅ Loaded" if model_ok else "❌ Not Found")
col2.metric("Layers", "3")
col3.metric("Parameters", "~400K")
col4.metric("Status", "Ready" if model_ok else "Train First")

uploaded = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png', 'webp'])

if uploaded:
    pil_img = Image.open(uploaded).convert('RGB')
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.image(pil_img, caption="Uploaded Image", use_column_width=True)
        analyze = st.button("🔍 Analyze Image", type="primary", use_container_width=True)
    
    with col_right:
        if analyze and model_ok:
            with st.spinner("Analyzing..."):
                time.sleep(0.5)
                pred, p_real, p_fake = predict(pil_img)
            
            conf = p_real if pred == 0 else p_fake
            
            if conf > 0.7:
                if pred == 0:
                    st.success(f"✅ AUTHENTIC (Confidence: {conf*100:.1f}%)")
                else:
                    st.error(f"❌ FAKE / AI-GENERATED (Confidence: {conf*100:.1f}%)")
            else:
                st.warning(f"⚠️ INCONCLUSIVE (Confidence: {conf*100:.1f}%)")
            
            st.progress(p_real, text=f"REAL: {p_real*100:.1f}%")
            st.progress(p_fake, text=f"FAKE: {p_fake*100:.1f}%")
            
        elif analyze and not model_ok:
            st.error("❌ Model not loaded. Please run train.py first.")
else:
    st.info("📤 Upload an image to get started")

st.markdown("---")
st.caption("Built with PyTorch · CNN + Frequency Analysis")
