"""
🔍 DeepFake & AI-Generated Image Detector
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
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
[data-testid="stProgress"]{background:#1e1e2e;}
</style>
""", unsafe_allow_html=True)

# ── النموذج (نفس اللي في train.py) ──────────────────────
class PixelCNN(nn.Module):
    """نموذج الكشف عن الصور المزيفة"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

@st.cache_resource
def load_model():
    """تحميل النموذج المدرب"""
    app_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(app_dir, 'pixel.pth')
    
    model = PixelCNN()
    
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            return model, True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return model, False
    
    return model, False

model, model_ok = load_model()

# ── Preprocessing ──────────────────────────────────────────
@st.cache_data
def load_mean_std():
    """تحميل قيم التطبيع"""
    app_dir = os.path.dirname(os.path.abspath(__file__))
    
    mean_path = os.path.join(app_dir, 'norm_mean.npy')
    std_path = os.path.join(app_dir, 'norm_std.npy')
    
    if os.path.exists(mean_path):
        mean = np.load(mean_path).reshape(3, 1, 1)
    else:
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    
    if os.path.exists(std_path):
        std = np.load(std_path).reshape(3, 1, 1)
    else:
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    
    return mean, std

MEAN, STD = load_mean_std()

def preprocess_image(pil_img):
    """تجهيز الصورة للنموذج"""
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return transform(pil_img).unsqueeze(0)

def predict(pil_img):
    """التنبؤ"""
    if not model_ok:
        return None, None, None
    
    tensor = preprocess_image(pil_img)
    
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1).squeeze().cpu().numpy()
    
    pred = int(probs.argmax())
    p_real = float(probs[0])
    p_fake = float(probs[1])
    
    return pred, p_real, p_fake

# ── UI ──────────────────────────────────────────────────────
st.title("🔍 DeepFake & AI Image Detector")
st.markdown("Upload an image to detect if it's REAL or AI-GENERATED")

# Stats
col1, col2, col3, col4 = st.columns(4)
col1.metric("Model", "✅ Loaded" if model_ok else "❌ Not Found")
col2.metric("Status", "Ready" if model_ok else "Run train.py")
col3.metric("Layers", "3")
col4.metric("Parameters", "~400K")

# Upload
uploaded = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png', 'webp'])

if uploaded:
    pil_img = Image.open(uploaded).convert('RGB')
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.image(pil_img, caption="Uploaded Image", use_column_width=True)
        analyze = st.button("🔍 Analyze Image", type="primary", use_container_width=True)
    
    with col_right:
        if analyze:
            if not model_ok:
                st.error("❌ Model not loaded. Please run `python train.py` first.")
            else:
                with st.spinner("Analyzing..."):
                    time.sleep(0.5)
                    pred, p_real, p_fake = predict(pil_img)
                
                if pred is not None:
                    conf = p_real if pred == 0 else p_fake
                    
                    if conf > 0.7:
                        if pred == 0:
                            st.success(f"✅ AUTHENTIC (Confidence: {conf*100:.1f}%)")
                        else:
                            st.error(f"❌ FAKE / AI-GENERATED (Confidence: {conf*100:.1f}%)")
                    else:
                        st.warning(f"⚠️ INCONCLUSIVE (Confidence: {conf*100:.1f}%)")
                    
                    # Probabilities
                    st.progress(p_real, text=f"REAL: {p_real*100:.1f}%")
                    st.progress(p_fake, text=f"FAKE: {p_fake*100:.1f}%")
                else:
                    st.error("❌ Prediction failed. Please check the model.")

else:
    st.info("📤 Upload an image to get started")

st.markdown("---")
st.caption("Built with PyTorch · CNN + Frequency Analysis")
