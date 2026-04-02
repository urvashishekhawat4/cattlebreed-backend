import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import copy
import time
import sys

# --- Configuration ---
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bcs dataaa", "bcs data"))
CSV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "bcs_labels.csv"))
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "bcs_model.pth")

# Hyperparameters for One-Cycle
BATCH_SIZE = 16
NUM_EPOCHS = 10  # Very fast for this small dataset
MAX_LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log(msg):
    print(msg)
    sys.stdout.flush()

# --- Custom Dataset ---
class BCSDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.label_map = {"fat": 0, "moderate": 1, "thin": 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label = self.label_map[self.data.iloc[idx, 1]]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# --- Data Augmentation ---
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Training Function ---
def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()
    best_acc = 0.0

    for epoch in range(num_epochs):
        log(f'Epoch {epoch + 1}/{num_epochs}')
        log('-' * 10)

        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            # Step the One-Cycle scheduler AFTER each batch update
            scheduler.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        log(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            log(f"--> Saved better weights (Acc: {best_acc:4f}) to {MODEL_SAVE_PATH}")

        log("")

    time_elapsed = time.time() - since
    log(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    log(f'Best Acc: {best_acc:4f}')
    return model

if __name__ == "__main__":
    log(f"Using device: {DEVICE}")
    
    # Check if data exists
    if not os.path.exists(DATA_DIR):
        log(f"ERROR: Data directory {DATA_DIR} not found.")
        sys.exit(1)
        
    dataset = BCSDataset(CSV_PATH, DATA_DIR, transform=train_transforms)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Use ResNet-18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # Initialize OneCycleLR scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 
                                              max_lr=MAX_LR, 
                                              steps_per_epoch=len(loader), 
                                              epochs=NUM_EPOCHS)

    train_model(model, loader, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS)
