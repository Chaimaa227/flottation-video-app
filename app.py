
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import base64
import io

app = Flask(__name__)
model = load_model("flottation_model.h5")
IMG_SIZE = 128

@app.route("/")
def index():
    return render_template("video.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    image_data = data['image']
    img_bytes = base64.b64decode(image_data.split(",")[1])
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)[0][0]
    label = "✅ Flottation efficace" if pred > 0.5 else "❌ Flottation inefficace"
    return jsonify({"prediction": label, "score": float(pred)})
