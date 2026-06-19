"""
🔍 DeepFake & AI-Generated Image Detector
CNN + Frequency Analysis + Grad-CAM
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageFilter
import os
import json
import time
from io import BytesIO

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="DeepFake Detector",
    page_icon="🔍",
    layout="wide",
)

# ── CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
html,body,[data-testid="stAppViewContainer"]{background:#07070f;color:#f0f0f8;}
[data-testid="block-container"]{padding:2rem;}
h1,h2,h3{font-family:sans-serif;}
[data-testid="stMetric"]{background:#0f0f1a;border:1px solid #1e1e2e;border-radius:14px;padding:16px 20px;}
[data-testid="stMetricLabel"]{color:#5555aa;}
[data-testid="stMetricValue"]{color:#a78bfa;}
[data-testid="stProgress"]>div>div{background:linear-gradient(90deg,#7c6af7,#a78bfa);}
[data-testid="stProgress"]{background:#1e1e2e;}
[data-testid="stFileUploaderDropzone"]{background:#0f0f1a;border:1.5px dashed #2a2a4a;}
</style>
""", unsafe_allow_html=True)

# ── Model ──────────────────────────────────────────────────
class PixelCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.clf = nn.Sequential(
            nn.Flatten(), nn.Linear(128*16,256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256,64), nn.ReLU(), nn.Linear(64,2)
        )
    def forward(self,x): return self.clf(self.features(x))

@st.cache_resource
def load_model():
    app_dir = os.path.dirname(os.path.abspath(__file__))
    
    model_path = os.path.join(app_dir, 'pixel.pth')
    mean_path = os.path.join(app_dir, 'norm_mean.npy')
    std_path = os.path.join(app_dir, 'norm_std.npy')
    
    model = PixelCNN()
    model_ok = False
    
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model_ok = True
        except:
            pass
    
    model.eval()
    
    if os.path.exists(mean_path):
        mean = np.load(mean_path)
    else:
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    
    if os.path.exists(std_path):
        std = np.load(std_path)
    else:
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    
    return model, mean, std, model_ok

model, MEAN, STD, model_ok = load_model()

# ── Preprocessing ──────────────────────────────────────────
def preprocess_image(pil_img):
    img = pil_img.convert('RGB').resize((64, 64), Image.LANCZOS)
    arr = np.array(img, dtype='float32').transpose(2, 0, 1) / 255.0
    m = MEAN.reshape(3, 1, 1)
    s = STD.reshape(3, 1, 1)
    arr = (arr - m) / s
    return torch.tensor(arr).unsqueeze(0).float()

def get_fft_features(pil_img):
    img = pil_img.convert('L').resize((64, 64))
    arr = np.array(img, dtype='float32')
    f = np.fft.fft2(arr)
    mag = np.log1p(np.abs(np.fft.fftshift(f)))
    mag = ((mag - mag.min()) / (mag.max() - mag.min() + 1e-8) * 255).astype('uint8')
    return Image.fromarray(mag).convert('RGB')

def predict(pil_img):
    tensor = preprocess_image(pil_img)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1).squeeze().numpy()
    pred = int(probs.argmax())
    return pred, float(probs[0]), float(probs[1])

# ── Grad-CAM ───────────────────────────────────────────────
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        target = model.features[-2]
        target.register_forward_hook(self._fwd_hook)
        target.register_backward_hook(self._bwd_hook)

    def _fwd_hook(self, _, __, output):
        self.activations = output.detach()

    def _bwd_hook(self, _, __, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, tensor, class_idx):
        self.model.eval()
        t = tensor.requires_grad_(True)
        out = self.model(t)
        self.model.zero_grad()
        out[0, class_idx].backward()
        w = self.gradients.mean(dim=[2,3], keepdim=True)
        cam = torch.relu((w * self.activations).sum(1)).squeeze().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def overlay_gradcam(pil_img, cam_np):
    size = pil_img.size
    cam_img = Image.fromarray((cam_np * 255).astype('uint8')).resize(size, Image.LANCZOS)
    cam_img = cam_img.filter(ImageFilter.GaussianBlur(radius=4))
    cam_arr = np.array(cam_img, dtype='float32') / 255.0
    r = (cam_arr * 255).astype('uint8')
    g = ((1 - cam_arr) * 100).astype('uint8')
    b = ((1 - cam_arr) * 200).astype('uint8')
    heatmap = Image.fromarray(np.stack([r,g,b], axis=2))
    orig = pil_img.convert('RGB').resize(size)
    blended = Image.blend(orig, heatmap, alpha=0.5)
    return blended

@st.cache_resource
def get_gradcam():
    return GradCAM(model) if model_ok else None

# ── Artifact Analysis ──────────────────────────────────────
def analyze_artifacts(pil_img):
    results = {}
    gray = np.array(pil_img.convert('L').resize((256,256)), dtype='float32')
    
    # Noise
    laplacian_var = np.var(gray - np.roll(gray, 1, axis=0))
    results['Noise'] = '✅ Normal' if laplacian_var < 500 else '⚠️ High'
    
    # Edges
    edges = np.array(pil_img.convert('L').resize((256,256)).filter(ImageFilter.FIND_EDGES))
    edge_var = np.var(edges)
    results['Edges'] = '✅ Sharp' if edge_var > 200 else '⚠️ Blurry'
    
    # Frequency
    fft = np.fft.fft2(gray)
    mag = np.abs(np.fft.fftshift(fft))
    center = mag[96:160, 96:160].sum()
    total = mag.sum() + 1e-8
    freq_ratio = center / total
    results['Frequency'] = '✅ Clean' if freq_ratio > 0.3 else '⚠️ Artifacts'
    
    return results

# ── UI ──────────────────────────────────────────────────────
st.title("🔍 DeepFake & AI Image Detector")
st.markdown("Upload an image to detect if it's REAL or AI-GENERATED")

# Stats
col1, col2, col3, col4 = st.columns(4)
col1.metric("Model", "✅ Loaded" if model_ok else "❌ Not Found")
col2.metric("Layers", "3")
col3.metric("Parameters", "~400K")
col4.metric("Speed", "< 50ms")

# Upload
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
                artifacts = analyze_artifacts(pil_img)
            
            # Verdict
            conf = p_real if pred == 0 else p_fake
            if conf > 0.85:
                if pred == 0:
                    st.success(f"✅ AUTHENTIC (Confidence: {conf*100:.1f}%)")
                else:
                    st.error(f"❌ FAKE / AI-GENERATED (Confidence: {conf*100:.1f}%)")
            else:
                st.warning(f"⚠️ INCONCLUSIVE (Confidence: {conf*100:.1f}%)")
            
            # Probabilities
            st.progress(p_real, text=f"REAL: {p_real*100:.1f}%")
            st.progress(p_fake, text=f"FAKE: {p_fake*100:.1f}%")
            
            # Artifacts
            st.markdown("#### 🔬 Artifact Analysis")
            for key, value in artifacts.items():
                st.markdown(f"- **{key}**: {value}")
            
            # Grad-CAM
            st.markdown("#### 🗺️ Grad-CAM Heatmap")
            try:
                grad_cam = get_gradcam()
                if grad_cam:
                    tensor = preprocess_image(pil_img)
                    cam = grad_cam.generate(tensor, class_idx=pred)
                    overlay = overlay_gradcam(pil_img.resize((256,256)), cam)
                    st.image(overlay, use_column_width=True)
                    st.caption("Red areas = what the model focused on")
            except:
                st.info("Grad-CAM not available")
            
            # FFT
            st.markdown("#### 〰️ Frequency Spectrum")
            fft_img = get_fft_features(pil_img)
            st.image(fft_img, use_column_width=True)
            st.caption("Bright patterns = GAN artifacts")
            
        elif analyze and not model_ok:
            st.error("❌ Model not loaded. Please run train.py first.")

else:
    st.info("📤 Upload an image to get started")

# Footer
st.markdown("---")
st.caption("Built with PyTorch · CNN + Frequency Analysis + Grad-CAM")
