import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

app = Flask(__name__)

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "model/best_model.pth"
NUM_CLASSES = 7

CLASS_NAMES = ["akiec","bcc","bkl","df","mel","nv","vasc"]

# -----------------------------
# BUILD MODEL (same as training)
# -----------------------------
def build_model():

    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

    return model


# -----------------------------
# LOAD MODEL
# -----------------------------
model = build_model()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=False))
model.eval()

# -----------------------------
# IMAGE TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def home():
    return jsonify({"message": "Skin Cancer Detection API Running"})


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    try:
        image = Image.open(file).convert("RGB")
    except:
        return jsonify({"error": "Invalid image"}), 400

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probs = F.softmax(outputs, dim=1)

        confidence, pred = torch.max(probs, 1)

    return jsonify({
        "prediction": CLASS_NAMES[pred.item()],
        "confidence": float(confidence.item())
    })


# -----------------------------
# RUN SERVER
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
