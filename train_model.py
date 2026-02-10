import os, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import models, transforms, datasets
import matplotlib.pyplot as plt
from tqdm import tqdm

# Settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "dataset"
MODEL_NAME = "mask_detector_v3.pth"

class TransformSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset, self.transform = subset, transform
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform: x = self.transform(x)
        return x, y
    def __len__(self): return len(self.subset)

def train():
    # 1. Pro Data Augmentation
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_ds = datasets.ImageFolder(root=DATA_DIR)
    print(f"[INFO] Found Classes: {full_ds.class_to_idx}")
    
    train_size = int(0.8 * len(full_ds))
    train_ds, val_ds = random_split(full_ds, [train_size, len(full_ds)-train_size])
    
    # Use small batch size for your small dataset (helps prevent overfitting)
    train_loader = DataLoader(TransformSubset(train_ds, train_tf), batch_size=4, shuffle=True)
    val_loader = DataLoader(TransformSubset(val_ds, val_tf), batch_size=4)

    # 2. Build Better Model
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    
    # Fine-tuning: Freeze 70% of the model, let the bottom 30% learn your face
    for i, param in enumerate(model.features.parameters()):
        if i < 80: param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.last_channel, 128),
        nn.ReLU(),
        nn.Linear(128, 2)
    )
    model.to(DEVICE)

    # LabelSmoothing helps when the dataset is very small
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4) # AdamW is more stable

    print(f"[START] Training on {DEVICE}...")
    for epoch in range(15):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for imgs, lbls in loop:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), MODEL_NAME)
    print(f"[DONE] Model saved as {MODEL_NAME}")

if __name__ == "__main__":
    train()