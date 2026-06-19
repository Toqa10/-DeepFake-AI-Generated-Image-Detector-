"""
train.py - تدريب نموذج DeepFake Detector
تحميل الداتا من Kaggle أونلاين + تقسيم Train/Validation/Test
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import kagglehub
from PIL import Image, ImageFilter, ImageEnhance
import random
from io import BytesIO

print("🚀 Starting training script...")

# ──────────────────────────────────────────────────────────────
# 1. تحميل الداتا من Kaggle
# ──────────────────────────────────────────────────────────────

def download_dataset():
    """تحميل الداتا من Kaggle مباشرة"""
    print("📥 Downloading dataset from Kaggle...")
    try:
        path = kagglehub.dataset_download("xhlulu/140k-real-and-fake-faces")
        print(f"✅ Dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f"⚠️ Could not download: {e}")
        print("   Using dummy data for testing...")
        return None

# ──────────────────────────────────────────────────────────────
# 2. معالجة الصور (للواقعية)
# ──────────────────────────────────────────────────────────────

class RealisticImageAugmenter:
    """تطبيق تشويش واقعي عشان النموذج يتعلم على صور حقيقية"""
    
    def __init__(self):
        self.augmentations = [
            self.add_jpeg_artifacts,
            self.add_noise,
            self.blur_slightly,
            self.change_brightness,
            self.change_contrast,
            self.add_crop_variation,
            self.add_color_shift,
        ]
    
    def add_jpeg_artifacts(self, img):
        quality = random.randint(70, 95)
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        return Image.open(buffer)
    
    def add_noise(self, img):
        img_array = np.array(img, dtype='float32') / 255.0
        noise = np.random.normal(0, random.uniform(0.005, 0.02), img_array.shape)
        img_array = np.clip(img_array + noise, 0, 1) * 255
        return Image.fromarray(img_array.astype('uint8'))
    
    def blur_slightly(self, img):
        radius = random.uniform(0.3, 1.2)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
    
    def change_brightness(self, img):
        factor = random.uniform(0.8, 1.2)
        return ImageEnhance.Brightness(img).enhance(factor)
    
    def change_contrast(self, img):
        factor = random.uniform(0.8, 1.2)
        return ImageEnhance.Contrast(img).enhance(factor)
    
    def add_crop_variation(self, img):
        w, h = img.size
        crop_factor = random.uniform(0.88, 0.98)
        new_w, new_h = int(w * crop_factor), int(h * crop_factor)
        left = random.randint(0, w - new_w)
        top = random.randint(0, h - new_h)
        cropped = img.crop((left, top, left + new_w, top + new_h))
        return cropped.resize((w, h), Image.LANCZOS)
    
    def add_color_shift(self, img):
        img_array = np.array(img)
        for channel in range(3):
            if random.random() < 0.3:
                shift = random.randint(-5, 5)
                img_array[:, :, channel] = np.clip(
                    img_array[:, :, channel] + shift, 0, 255
                )
        return Image.fromarray(img_array.astype('uint8'))
    
    def augment(self, img, is_fake=False):
        if is_fake:
            num_augs = random.randint(2, 5)
        else:
            num_augs = random.randint(1, 3)
        
        selected = random.sample(self.augmentations, min(num_augs, len(self.augmentations)))
        
        for aug in selected:
            if random.random() < 0.6:
                try:
                    img = aug(img)
                except:
                    pass
        return img

# ──────────────────────────────────────────────────────────────
# 3. تحضير البيانات مع Augmentation
# ──────────────────────────────────────────────────────────────

class RealisticDataset:
    """كلاس مخصص لقراءة الصور مع تطبيق التشويش"""
    
    def __init__(self, data_path, transform=None, augment=True):
        self.data_path = data_path
        self.transform = transform
        self.augment = augment
        self.augmenter = RealisticImageAugmenter()
        self.images = []
        self.labels = []
        
        # قراءة الصور من مجلدات real و fake
        for label, folder in enumerate(['real', 'fake']):
            folder_path = os.path.join(data_path, folder)
            if os.path.exists(folder_path):
                for img_name in os.listdir(folder_path):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.images.append(os.path.join(folder_path, img_name))
                        self.labels.append(label)
        
        print(f"📊 Loaded {len(self.images)} images")
        print(f"   REAL: {self.labels.count(0)}, FAKE: {self.labels.count(1)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            img = Image.new('RGB', (64, 64), color='black')
        
        # تطبيق التشويش
        if self.augment:
            is_fake = (label == 1)
            img = self.augmenter.augment(img, is_fake=is_fake)
        
        img = img.resize((64, 64), Image.LANCZOS)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

def get_data_loaders(data_path, batch_size=64, use_dummy=False):
    """
    تحضير البيانات وتقسيمها إلى Train / Validation / Test
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # ── استخدام داتا حقيقية ──
    if data_path and os.path.exists(data_path) and not use_dummy:
        print("📊 Using real dataset...")
        dataset = RealisticDataset(data_path, transform=transform, augment=True)
        
        if len(dataset) == 0:
            print("⚠️ No images found, using dummy data")
            return get_dummy_loaders(batch_size)
    
    # ── داتا وهمية للاختبار ──
    print("📊 Using dummy data for testing...")
    return get_dummy_loaders(batch_size)

def get_dummy_loaders(batch_size):
    """عمل داتا وهمية عشان نجرب الكود"""
    num_train, num_val, num_test = 800, 100, 100
    
    train_data = torch.randn(num_train, 3, 64, 64)
    train_labels = torch.randint(0, 2, (num_train,))
    
    val_data = torch.randn(num_val, 3, 64, 64)
    val_labels = torch.randint(0, 2, (num_val,))
    
    test_data = torch.randn(num_test, 3, 64, 64)
    test_labels = torch.randint(0, 2, (num_test,))
    
    from torch.utils.data import TensorDataset
    
    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    test_dataset = TensorDataset(test_data, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

# ──────────────────────────────────────────────────────────────
# 4. تعريف النموذج
# ──────────────────────────────────────────────────────────────

class PixelCNN(nn.Module):
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
        self.clf = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(128 * 16, 256), 
            nn.ReLU(), 
            nn.Dropout(0.4),
            nn.Linear(256, 64), 
            nn.ReLU(), 
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        return self.clf(self.features(x))

# ──────────────────────────────────────────────────────────────
# 5. التدريب والتقييم
# ──────────────────────────────────────────────────────────────

def evaluate_model(model, test_loader, device):
    """تقييم النموذج على مجموعة الاختبار"""
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total

def train_model(epochs=10, batch_size=64):
    """
    التدريب الرئيسي
    """
    print("🚀 Starting training...")
    
    # ── 1. تحميل الداتا ──
    data_path = download_dataset()
    train_loader, val_loader, test_loader = get_data_loaders(
        data_path, batch_size, use_dummy=(data_path is None)
    )
    
    if train_loader is None:
        print("❌ Failed to load data")
        return
    
    # ── 2. إعداد النموذج ──
    model = PixelCNN()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print(f"📊 Device: {device}")
    print(f"📊 Training samples: {len(train_loader.dataset)}")
    print(f"📊 Validation samples: {len(val_loader.dataset)}")
    print(f"📊 Test samples: {len(test_loader.dataset)}")
    
    # ── 3. التدريب ──
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0
    
    for epoch in range(epochs):
        # ── التدريب ──
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        
        # ── التحقق (Validation) ──
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        # ── حفظ أفضل نموذج ──
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('app', exist_ok=True)
            torch.save(model.state_dict(), 'app/pixel.pth')
            print(f"⭐ New best model! Validation Acc: {val_acc:.2f}%")
        
        # ── تسجيل التاريخ ──
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        scheduler.step()
        
        print(f'Epoch {epoch+1}:')
        print(f'  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
    
    # ── 4. الاختبار النهائي ──
    print("\n🧪 Final Testing...")
    model.load_state_dict(torch.load('app/pixel.pth', map_location='cpu'))
    test_acc = evaluate_model(model, test_loader, device)
    print(f"✅ Final Test Accuracy: {test_acc:.2f}%")
    
    # ── 5. حفظ النتائج ──
    results = {
        'accuracy': test_acc / 100,
        'best_val': best_val_acc / 100,
        'history': history
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("📊 Results saved to results/results.json")
    
    # ── 6. حفظ MEAN و STD للاستخدام في التطبيق ──
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    np.save('app/norm_mean.npy', mean)
    np.save('app/norm_std.npy', std)
    print("✅ Normalization params saved to app/")
    
    return model, history, test_acc

# ──────────────────────────────────────────────────────────────
# 6. التشغيل
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_model(epochs=10, batch_size=64)
