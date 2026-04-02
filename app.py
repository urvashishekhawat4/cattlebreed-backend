import os
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder="../frontend")
CORS(app)

BREED_CLASSES = [
    "Bhadawari", "Gir", "Hariana", "Kankrej", "Mehsana", "Murrah",
    "Nagori", "Rathi", "Red_Sindhi", "Sahiwal", "Surti", "Tharparkar",
]
BCS_CLASSES = ["fat", "moderate", "thin"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

breed_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

bcs_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def _load_resnet18(num_classes: int, weights_path: str):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"WARNING: {weights_path} not found — model will return random predictions")
    model.to(device)
    model.eval()
    return model


breed_model = _load_resnet18(len(BREED_CLASSES), "rajasthan_cattle_modell.pth")
bcs_model = _load_resnet18(len(BCS_CLASSES), "bcs_model.pth")


def _predict(model, transform, class_names, image: Image.Image):
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)
        confidence, idx = torch.max(probs, 1)
    return class_names[idx.item()], round(confidence.item() * 100, 2)


@app.route("/")
def index():
    return send_from_directory("../frontend", "index.html")


@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory("../frontend", path)


@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    breed, breed_conf = _predict(breed_model, breed_transform, BREED_CLASSES, img)
    bcs, bcs_conf = _predict(bcs_model, bcs_transform, BCS_CLASSES, img)

    return jsonify({
        "breed": breed,
        "breed_confidence": breed_conf,
        "bcs": bcs,
        "bcs_confidence": bcs_conf,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
