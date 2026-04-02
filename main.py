import os
import io
import datetime
import bcrypt
import jwt
import uuid
import shutil
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from pydantic import BaseModel

# Import our new database models
from models import SessionLocal, User, PredictionLog, init_db

# --- Configuration ---
BREED_CLASSES = [
    "Bhadawari", "Gir", "Hariana", "Kankrej", "Mehsana", "Murrah",
    "Nagori", "Rathi", "Red_Sindhi", "Sahiwal", "Surti", "Tharparkar",
]
BCS_CLASSES = ["fat", "moderate", "thin"]

JWT_SECRET = "super-secret-key-for-dev-only"  # Should be in .env for production
JWT_ALGORITHM = "HS256"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Database & Auth Dependencies ---
init_db()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(authorization: str = Header(None), db: Session = Depends(get_db)):
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("user_id")
        return db.query(User).filter(User.id == user_id).first()
    except Exception:
        return None

# --- Deep Learning Model Logic ---
breed_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
bcs_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def _load_resnet18(num_classes: int, weights_name: str):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    path = os.path.join(os.path.dirname(__file__), weights_name)
    if os.path.exists(path):
        try:
            state = torch.load(path, map_location=device, weights_only=True)
            model.load_state_dict(state)
        except Exception:
            state = torch.load(path, map_location=device)
            model.load_state_dict(state)
        print(f"Loaded weights from {path}")
    else:
        print(f"WARNING: {weights_name} not found — model will serve randomly initialized predictions.")
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

# --- FastAPI Implementation ---
app = FastAPI(title="Cattle Breed Analyzer API with Auth")

uploads_dir = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(uploads_dir, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=uploads_dir), name="uploads")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AuthSchema(BaseModel):
    email: str
    password: str

class RegisterSchema(BaseModel):
    name: str
    email: str
    password: str

class ProfileSchema(BaseModel):
    name: str

@app.post("/api/register")
def register_user(data: RegisterSchema, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == data.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_pw = bcrypt.hashpw(data.password.encode('utf-8'), bcrypt.gensalt())
    new_user = User(name=data.name, email=data.email, password_hash=hashed_pw.decode('utf-8'))
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User successfully registered!"}

@app.post("/api/login")
def login_user(data: AuthSchema, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email).first()
    if not user or not bcrypt.checkpw(data.password.encode('utf-8'), user.password_hash.encode('utf-8')):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    token = jwt.encode(
        {"user_id": user.id, "exp": datetime.datetime.utcnow() + datetime.timedelta(days=7)},
        JWT_SECRET,
        algorithm=JWT_ALGORITHM
    )
    return {"token": token, "email": user.email, "name": user.name, "profile_picture_url": user.profile_picture_url}

@app.put("/api/profile")
def update_profile(data: ProfileSchema, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    current_user.name = data.name
    db.commit()
    return {"message": "Profile updated", "name": current_user.name, "profile_picture_url": current_user.profile_picture_url}

@app.post("/api/profile/picture")
def upload_profile_picture(image: UploadFile = File(...), db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    ext = os.path.splitext(image.filename)[1]
    filename = f"user_{current_user.id}_{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(uploads_dir, filename)
    
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
        
    current_user.profile_picture_url = f"http://localhost:5000/uploads/{filename}"
    db.commit()
    
    return {"message": "Profile picture updated", "profile_picture_url": current_user.profile_picture_url}

@app.get("/api/history")
def get_history(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    logs = db.query(PredictionLog).filter(PredictionLog.user_id == current_user.id).order_by(PredictionLog.timestamp.desc()).all()
    return logs

@app.delete("/api/history")
def clear_all_history(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    db.query(PredictionLog).filter(PredictionLog.user_id == current_user.id).delete()
    db.commit()
    return {"message": "All history cleared"}

@app.delete("/api/history/{log_id}")
def delete_history_item(log_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    log = db.query(PredictionLog).filter(PredictionLog.id == log_id, PredictionLog.user_id == current_user.id).first()
    if not log:
        raise HTTPException(status_code=404, detail="Log not found")
    db.delete(log)
    db.commit()
    return {"message": "History item deleted"}

@app.post("/api/predict")
async def predict(image: UploadFile = File(...), db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    breed, breed_conf = _predict(breed_model, breed_transform, BREED_CLASSES, img)
    bcs, bcs_conf = _predict(bcs_model, bcs_transform, BCS_CLASSES, img)

    # Log the request natively into SQLite if user is logged in
    if current_user:
        log_entry = PredictionLog(
            user_id=current_user.id,
            image_filename=image.filename,
            breed_result=breed,
            breed_confidence=breed_conf,
            bcs_result=bcs,
            bcs_confidence=bcs_conf,
        )
        db.add(log_entry)
        db.commit()

    return {
        "breed": breed,
        "breed_confidence": breed_conf,
        "bcs": bcs,
        "bcs_confidence": bcs_conf,
        "logged_to_history": bool(current_user)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
