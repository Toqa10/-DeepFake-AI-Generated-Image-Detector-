# 🔍 DeepFake & AI-Generated Image Detector

> **CNN + Frequency Analysis + Grad-CAM — Phase 2 Project**

---

## 🎯 What it does

Upload any image → the system tells you if it's **REAL** or **AI-GENERATED** using 3 parallel analysis streams:

| Stream | Method | What it detects |
|--------|--------|-----------------|
| 🧠 Pixel CNN | 3-block CNN | Texture artifacts, biological inconsistencies |
| 〰️ FFT Analysis | Frequency domain | GAN upsampling periodic signatures |
| 🗺️ Grad-CAM | Gradient visualization | Exactly which region triggered detection |

---

## 🚀 Run it

```bash
pip install -r requirements.txt
python train.py          # train the model
cd app
streamlit run app.py     # launch the app
```

---

## 🏗️ Architecture

```
Input (64×64×3)
    ↓
Block 1: Conv(3→32) + BN + ReLU + MaxPool
Block 2: Conv(32→64) + BN + ReLU + MaxPool  
Block 3: Conv(64→128) + BN + ReLU + AvgPool
    ↓
FC: 2048 → 256 → 64 → 2
    ↓
Softmax → REAL / FAKE
```

---

## 📊 Results

- **Test Accuracy:** 100% (on synthetic dataset)
- **Parameters:** ~400K
- **Inference:** < 50ms

---

## 🔬 Why DeepFakes are detectable

GANs use **Transposed Convolution** to upsample noise → full image.
This creates periodic artifacts in the **frequency domain** (FFT) — a fingerprint every GAN leaves behind.

Real images have smooth, natural frequency spectra.
AI-generated images show unusual bright rings or grid patterns.

---

*Phase 2 of AI/ML Learning Roadmap — CNN + Computer Vision*
