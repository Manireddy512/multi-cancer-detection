from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained model
model = tf.keras.models.load_model("model/best_model.h5")

def preprocess_image(image_path):
    img = Image.open(image_path).resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/", methods=["GET","POST"])
def index():
    prediction = ""

    if request.method == "POST":
        file = request.files["image"]
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        img = preprocess_image(filepath)

        pred = model.predict(img)

        if pred[0][0] > 0.5:
            prediction = "Cancer Detected"
        else:
            prediction = "No Cancer Detected"

    return render_template("index.html", prediction=prediction)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
