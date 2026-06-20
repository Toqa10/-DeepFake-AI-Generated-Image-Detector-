import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, classification_report
import seaborn as sns, os, json, time

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
device = torch.device('cpu')
print(f"🖥️  Device: {device}")

# ── Data — يقرأ من الـ Kaggle folders مباشرة ─────────────
from torchvision import datasets, transforms

DATA_DIR = 'data/real_vs_fake'
if not os.path.exists(DATA_DIR):
    print("❌ الداتا مش موجودة!")
    print("حطي الداتا هنا: data/real_vs_fake/train/real & fake")
    exit(1)

train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

train_ds = datasets.ImageFolder(os.path.join(DATA_DIR,'train'), train_transform)
val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR,'valid'), val_transform)

print(f"\n📂 Class mapping: {train_ds.class_to_idx}")
FAKE_IDX = train_ds.class_to_idx.get('fake', 0)
REAL_IDX = train_ds.class_to_idx.get('real', 1)

tr_ld = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=2)
va_ld = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=2)
te_ld = va_ld  # نستخدم validation كـ test

print(f"✅ Train:{len(train_ds):,}  Val:{len(val_ds):,}")

# ImageNet normalization — ثابتة مش محتاجة تتحسب
np.save('data/norm_mean.npy', np.array([0.485,0.456,0.406],'float32').reshape(3,1,1))
np.save('data/norm_std.npy',  np.array([0.229,0.224,0.225],'float32').reshape(3,1,1))

def fft_features_batch(batch):
    """بتاخد batch tensor وبترجع FFT magnitude"""
    B,C,H,W = batch.shape
    out = torch.zeros_like(batch)
    imgs = batch.cpu().numpy()
    for i in range(B):
        for c in range(C):
            f   = np.fft.fft2(imgs[i,c])
            mag = np.log1p(np.abs(np.fft.fftshift(f))).astype('float32')
            mn,mx = mag.min(), mag.max()
            out[i,c] = torch.tensor((mag-mn)/(mx-mn+1e-8))
    return out.to(batch.device)

# ── Models ────────────────────────────────────────────────
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
            nn.Flatten(), nn.Linear(256*16,512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256,64), nn.ReLU(), nn.Linear(64,2)
        )
    def forward(self,x): return self.clf(self.features(x))

class FreqCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.clf = nn.Sequential(
            nn.Flatten(), nn.Linear(64*16,128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128,2)
        )
    def forward(self,x): return self.clf(self.features(x))

class Ensemble(nn.Module):
    def __init__(self,p,f):
        super().__init__()
        self.pixel=p; self.freq=f
        self.fusion=nn.Sequential(nn.Linear(4,32),nn.ReLU(),nn.Linear(32,2))
    def forward(self,xp,xf):
        return self.fusion(torch.cat([self.pixel(xp),self.freq(xf)],dim=1))

pixel_m = PixelCNN().to(device)
freq_m  = FreqCNN().to(device)

# ── Train ─────────────────────────────────────────────────
def train_one(model, use_freq, save_path, epochs=15, name=""):
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    opt  = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch  = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    hist = {'tl':[],'ta':[],'vl':[],'va':[]}
    best = 0
    print(f"\n🚀 {name}  ({epochs} epochs)")
    print(f"{'Ep':>3}|{'TrL':>7}|{'TrA':>6}|{'VaL':>7}|{'VaA':>6}")
    print("─"*36)
    for ep in range(1,epochs+1):
        model.train()
        tl=tc=tt=0
        for imgs,yb in tr_ld:
            imgs,yb=imgs.to(device),yb.to(device)
            xin=fft_features_batch(imgs) if use_freq else imgs
            out=model(xin)
            loss=crit(out,yb)
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
            tl+=loss.item()*yb.size(0);tc+=(out.argmax(1)==yb).sum().item();tt+=yb.size(0)
        sch.step()
        model.eval(); vl=vc=vt=0
        with torch.no_grad():
            for imgs,yb in va_ld:
                imgs,yb=imgs.to(device),yb.to(device)
                xin=fft_features_batch(imgs) if use_freq else imgs
                out=model(xin);loss=crit(out,yb)
                vl+=loss.item()*yb.size(0);vc+=(out.argmax(1)==yb).sum().item();vt+=yb.size(0)
        ta,va_=100*tc/tt,100*vc/vt
        hist['tl'].append(tl/tt);hist['ta'].append(ta)
        hist['vl'].append(vl/vt);hist['va'].append(va_)
        star=''
        if va_>best: best=va_; torch.save(model.state_dict(),save_path); star=' 💾'
        if ep%5==0 or ep==1:
            print(f"{ep:>3}|{tl/tt:>7.4f}|{ta:>5.1f}%|{vl/vt:>7.4f}|{va_:>5.1f}%{star}")
    print(f"✅ Best: {best:.2f}%")
    return hist, best

h_px, b_px = train_one(pixel_m, False, 'models/pixel.pth', 20, "PixelCNN")
h_fx, b_fx = train_one(freq_m,  True,  'models/freq.pth',  20, "FreqCNN")

pixel_m.load_state_dict(torch.load('models/pixel.pth', map_location=device, weights_only=True))
freq_m.load_state_dict( torch.load('models/freq.pth',  map_location=device, weights_only=True))

# Ensemble
print("\n🚀 Ensemble fusion...")
ens = Ensemble(pixel_m, freq_m).to(device)
for p in ens.pixel.parameters(): p.requires_grad=False
for p in ens.freq.parameters():  p.requires_grad=False
ens_opt  = optim.Adam(ens.fusion.parameters(), lr=5e-3)
ens_crit = nn.CrossEntropyLoss()
best_ens = 0
for ep in range(15):
    ens.train()
    for imgs,yb in tr_ld:
        imgs,yb=imgs.to(device),yb.to(device)
        fft_imgs=fft_features_batch(imgs)
        loss=ens_crit(ens(imgs,fft_imgs),yb)
        ens_opt.zero_grad();loss.backward();ens_opt.step()
    ens.eval(); vc=vt=0
    with torch.no_grad():
        for imgs,yb in va_ld:
            imgs,yb=imgs.to(device),yb.to(device)
            out=ens(imgs,fft_features_batch(imgs));vc+=(out.argmax(1)==yb).sum().item();vt+=yb.size(0)
    va_=100*vc/vt
    if va_>best_ens: best_ens=va_; torch.save(ens.state_dict(),'models/ensemble.pth')
    if (ep+1)%5==0: print(f"   Ep {ep+1:>2}: {va_:.2f}%")

ens.load_state_dict(torch.load('models/ensemble.pth', map_location=device, weights_only=True))
print(f"✅ Ensemble Best: {best_ens:.2f}%")

# ── Test Evaluation ───────────────────────────────────────
print("\n📊 Test Evaluation...")
ens.eval()
preds=[]; probs=[]; labels=[]
with torch.no_grad():
    for imgs,yb in te_ld:
        imgs=imgs.to(device)
        fft_imgs=fft_features_batch(imgs)
        out=ens(imgs,fft_imgs)
        pr=torch.softmax(out,1)
        preds.extend(out.argmax(1).cpu().numpy())
        probs.extend(pr[:,FAKE_IDX].cpu().numpy())
        labels.extend(yb.numpy())
preds=np.array(preds); probs=np.array(probs); labels=np.array(labels)
acc = (preds==labels).mean()*100
auc = roc_auc_score(labels,probs)
print(f"🎯 Accuracy: {acc:.2f}%  AUC: {auc:.4f}")
print(classification_report(labels,preds,target_names=['REAL','FAKE']))

# ── Charts ─────────────────────────────────────────────────
fig,axes=plt.subplots(2,3,figsize=(18,10))
fig.suptitle('DeepFake Detector — Results',fontsize=14,fontweight='bold')
eps_p=range(1,len(h_px['tl'])+1); eps_f=range(1,len(h_fx['tl'])+1)
for ax,hist,name in [(axes[0,0],h_px,'PixelCNN'),(axes[1,0],h_fx,'FreqCNN')]:
    ep=range(1,len(hist['tl'])+1)
    ax.plot(ep,hist['tl'],'b-o',ms=3,label='Train'); ax.plot(ep,hist['vl'],'r-o',ms=3,label='Val')
    ax.set_title(f'{name} Loss'); ax.legend(); ax.grid(alpha=.3)
for ax,hist,name in [(axes[0,1],h_px,'PixelCNN'),(axes[1,1],h_fx,'FreqCNN')]:
    ep=range(1,len(hist['ta'])+1)
    ax.plot(ep,hist['ta'],'b-o',ms=3,label='Train'); ax.plot(ep,hist['va'],'r-o',ms=3,label='Val')
    ax.set_title(f'{name} Accuracy'); ax.legend(); ax.grid(alpha=.3)
fpr,tpr,_=roc_curve(labels,probs)
axes[0,2].plot(fpr,tpr,'b-',lw=2,label=f'AUC={auc:.3f}')
axes[0,2].plot([0,1],[0,1],'r--',alpha=.5); axes[0,2].set_title('ROC Curve')
axes[0,2].legend(); axes[0,2].grid(alpha=.3)
cm=confusion_matrix(labels,preds)
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',ax=axes[1,2],
            xticklabels=['REAL','FAKE'],yticklabels=['REAL','FAKE'])
axes[1,2].set_title('Confusion Matrix')
plt.tight_layout(); plt.savefig('results/training.png',dpi=140,bbox_inches='tight'); plt.close()
print("💾 results/training.png")

# Confidence
fig,ax=plt.subplots(figsize=(10,5))
ax.hist(probs[labels==0],bins=30,color='green',alpha=.7,label='REAL')
ax.hist(probs[labels==1],bins=30,color='red',  alpha=.7,label='FAKE')
ax.set_title('Confidence Distribution'); ax.legend(); ax.grid(alpha=.3)
ax.set_xlabel('P(FAKE)'); ax.set_ylabel('Count')
plt.tight_layout(); plt.savefig('results/confidence.png',dpi=140,bbox_inches='tight'); plt.close()
print("💾 results/confidence.png")

json.dump(dict(accuracy=round(acc,4),auc=round(auc,4),
               pixel_best=round(b_px,4),freq_best=round(b_fx,4),
               ensemble_best=round(best_ens,4)),
          open('results/results.json','w'),indent=2)

print(f"\n{'═'*45}")
print(f"🎉 Done!  Accuracy:{acc:.2f}%  AUC:{auc:.4f}")
print(f"{'═'*45}")
