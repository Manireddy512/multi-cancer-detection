import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, render_template

app = Flask(__name__)

# Load model
model = torch.load("model/best_model.pth", map_location=torch.device("cpu"))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

@app.route("/", methods=["GET","POST"])
def index():
    prediction = ""

    if request.method == "POST":
        file = request.files["image"]

        img = Image.open(file).convert("RGB")
        img = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img)
            pred = torch.argmax(output, dim=1)

        prediction = f"Prediction: {pred.item()}"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
