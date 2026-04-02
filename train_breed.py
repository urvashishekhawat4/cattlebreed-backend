import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import copy
import time
import sys

# --- Configuration ---
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rajasthan_cattle_dataset", "rajasthan_cattle_dataset"))
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "rajasthan_cattle_modell.pth")

# Hyperparameters for One-Cycle
BATCH_SIZE = 32
NUM_EPOCHS = 11  # Resuming for remaining 11 cycles
MAX_LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log(msg):
    print(msg)
    sys.stdout.flush()

# --- Data Augmentation ---
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Load Datasets ---
def load_data():
    log(f"Loading Breed data from {TRAIN_DIR}...")
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader, len(train_dataset.classes)

# --- Training Function ---
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=15):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    
    # Evaluate initial val accuracy so we don't overwrite good weights with a bad Epoch 0
    model.eval()
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
    best_acc = (running_corrects.double() / len(val_loader.dataset)).item()
    log(f"Baseline Val Acc before resuming: {best_acc:.4f}")

    for epoch in range(num_epochs):
        log(f'Epoch {epoch + 1}/{num_epochs}')
        log('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        # Step the One-Cycle scheduler AFTER each batch update
                        scheduler.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            log(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    
                    # Safe saving: write to a .tmp file first, then replace atomically to avoid corruption on abrupt termination
                    temp_save_path = MODEL_SAVE_PATH + ".tmp"
                    torch.save(best_model_wts, temp_save_path)
                    os.replace(temp_save_path, MODEL_SAVE_PATH)
                    
                    log(f"--> Saved better weights (Breed Acc: {best_acc:4f}) to {MODEL_SAVE_PATH}")

        log("")

    time_elapsed = time.time() - since
    log(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    log(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model

# --- Main Execution ---
if __name__ == "__main__":
    log(f"Using device: {DEVICE}")
    
    train_loader, test_loader, num_classes = load_data()
    
    # Initialize Model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    if os.path.exists(MODEL_SAVE_PATH):
        try:
            log(f"--> Found existing weights at {MODEL_SAVE_PATH}, resuming...")
            model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        except Exception as e:
            log(f"--> Could not load saved weights: {e}")

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001) # Initial LR for Adam
    
    # Initialize OneCycleLR scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 
                                              max_lr=MAX_LR, 
                                              steps_per_epoch=len(train_loader), 
                                              epochs=NUM_EPOCHS)

    model = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS)
