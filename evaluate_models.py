import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Breed Evaluation ---
BREED_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rajasthan_cattle_dataset", "rajasthan_cattle_dataset"))
BREED_TEST_DIR = os.path.join(BREED_DATA_DIR, "test")
BREED_MODEL_PATH = os.path.join(os.path.dirname(__file__), "rajasthan_cattle_modell.pth")
BREED_CLASSES = [
    "Bhadawari", "Gir", "Hariana", "Kankrej", "Mehsana", "Murrah",
    "Nagori", "Rathi", "Red_Sindhi", "Sahiwal", "Surti", "Tharparkar",
]

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def evaluate_breed():
    print("Evaluating Breed Model...")
    if not os.path.exists(BREED_TEST_DIR):
        print(f"Test dir not found: {BREED_TEST_DIR}")
        return

    test_dataset = datasets.ImageFolder(BREED_TEST_DIR, transform=val_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(BREED_CLASSES))
    try:
        model.load_state_dict(torch.load(BREED_MODEL_PATH, map_location=device))
    except Exception:
        state = torch.load(BREED_MODEL_PATH, map_location=device)
        model.load_state_dict(state)
        
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Print Class Report
    report = classification_report(all_labels, all_preds, target_names=test_dataset.classes, output_dict=False)
    print("--- Breed Classification Report ---")
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Breed Model')
    plt.tight_layout()
    plt.savefig('breed_confusion_matrix.png')
    plt.close()
    print("Saved breed_confusion_matrix.png")

# --- BCS Evaluation ---
BCS_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bcs dataaa", "bcs data"))
BCS_CSV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "bcs_labels.csv"))
BCS_MODEL_PATH = os.path.join(os.path.dirname(__file__), "bcs_model.pth")
BCS_CLASSES = ["fat", "moderate", "thin"]

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

def evaluate_bcs():
    print("\nEvaluating BCS Model...")
    if not os.path.exists(BCS_DATA_DIR):
        print(f"Data dir not found: {BCS_DATA_DIR}")
        return

    dataset = BCSDataset(BCS_CSV_PATH, BCS_DATA_DIR, transform=val_transforms)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(BCS_CLASSES))
    try:
        model.load_state_dict(torch.load(BCS_MODEL_PATH, map_location=device))
    except Exception:
        state = torch.load(BCS_MODEL_PATH, map_location=device)
        model.load_state_dict(state)

    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    target_names = ["fat", "moderate", "thin"]

    report = classification_report(all_labels, all_preds, target_names=target_names, output_dict=False)
    print("--- BCS Classification Report ---")
    print(report)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - BCS Model')
    plt.tight_layout()
    plt.savefig('bcs_confusion_matrix.png')
    plt.close()
    print("Saved bcs_confusion_matrix.png")

if __name__ == "__main__":
    evaluate_breed()
    evaluate_bcs()
