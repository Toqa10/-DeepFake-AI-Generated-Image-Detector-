"""
train.py - تدريب نموذج DeepFake Detector
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
import random
from PIL import Image

print("🚀 Starting training...")

# ──────────────────────────────────────────────────────────────
# 1. تعريف النموذج (نفس اللي في app.py)
# ──────────────────────────────────────────────────────────────

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

# ──────────────────────────────────────────────────────────────
# 2. تحميل الداتا
# ──────────────────────────────────────────────────────────────

def get_data_loaders(data_path=None, batch_size=64):
    """تحميل وتقسيم البيانات"""
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # ── محاولة تحميل داتا حقيقية ──
    if data_path and os.path.exists(data_path):
        try:
            dataset = datasets.ImageFolder(data_path, transform=transform)
            if len(dataset) > 0:
                print(f"📊 Loaded {len(dataset)} real images")
                return split_dataset(dataset, batch_size)
        except:
            print("⚠️ Error loading real data")
    
    # ── داتا وهمية للاختبار ──
    print("📊 Using dummy data for testing...")
    return create_dummy_data(batch_size)

def split_dataset(dataset, batch_size):
    """تقسيم البيانات"""
    total = len(dataset)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size
    
    train_data, val_data, test_data = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    print(f"📊 Train: {train_size}, Val: {val_size}, Test: {test_size}")
    return train_loader, val_loader, test_loader

def create_dummy_data(batch_size):
    """داتا وهمية مع فروق واضحة"""
    from torch.utils.data import TensorDataset
    
    # REAL: صور فاتحة
    real = torch.randn(2000, 3, 64, 64) * 0.2 + 0.8
    real = torch.clamp(real, 0, 1)
    real_labels = torch.zeros(2000)
    
    # FAKE: صور داكنة
    fake = torch.randn(2000, 3, 64, 64) * 0.2 + 0.2
    fake = torch.clamp(fake, 0, 1)
    fake_labels = torch.ones(2000)
    
    images = torch.cat([real, fake])
    labels = torch.cat([real_labels, fake_labels])
    
    idx = torch.randperm(4000)
    images, labels = images[idx], labels[idx]
    
    dataset = TensorDataset(images, labels)
    return split_dataset(dataset, batch_size)

# ──────────────────────────────────────────────────────────────
# 3. التدريب
# ──────────────────────────────────────────────────────────────

def train_model(data_path=None, epochs=15, batch_size=64):
    """التدريب الرئيسي"""
    print("🚀 Starting training...")
    
    # ── تحميل الداتا ──
    train_loader, val_loader, test_loader = get_data_loaders(data_path, batch_size)
    
    # ── إعداد النموذج ──
    model = PixelCNN()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    
    print(f"📊 Device: {device}")
    print(f"📊 Training on {len(train_loader.dataset)} samples")
    
    # ── التدريب ──
    best_acc = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # تدريب
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
        
        # تقييم
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
        scheduler.step(val_loss)
        
        # حفظ التاريخ
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # حفظ أفضل نموذج
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs('app', exist_ok=True)
            torch.save(model.state_dict(), 'app/pixel.pth')
            print(f"⭐ New best model! Val Acc: {val_acc:.2f}%")
        
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
    
    # ── اختبار نهائي ──
    model.load_state_dict(torch.load('app/pixel.pth', map_location='cpu'))
    model.eval()
    
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    
    test_acc = 100. * test_correct / test_total
    print(f"\n✅ Final Test Accuracy: {test_acc:.2f}%")
    
    # ── حفظ النتائج ──
    results = {
        'accuracy': test_acc / 100,
        'best_val': best_acc / 100,
        'history': history
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # حفظ MEAN/STD
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    np.save('app/norm_mean.npy', mean)
    np.save('app/norm_std.npy', std)
    
    print("✅ Model saved to app/pixel.pth")
    print("✅ Normalization params saved to app/")
    print("📊 Results saved to results/results.json")

# ──────────────────────────────────────────────────────────────
# 4. التشغيل
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # حاول تحميل الداتا من Kaggle
    data_path = None
    
    try:
        import kagglehub
        print("📥 Downloading dataset from Kaggle...")
        data_path = kagglehub.dataset_download("xhlulu/140k-real-and-fake-faces")
        print(f"✅ Dataset downloaded to: {data_path}")
    except:
        print("⚠️ Could not download dataset, using dummy data")
    
    train_model(data_path=data_path, epochs=15, batch_size=64)
