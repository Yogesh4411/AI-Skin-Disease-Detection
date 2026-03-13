# ==========================================
# AI SKIN DISEASE DETECTION - FLASK API
# ==========================================

from flask import Flask, request, jsonify
import os
from predict import predict_disease

app = Flask(__name__)

# Folder to store uploaded images
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ==========================================
# HOME ROUTE
# ==========================================

@app.route("/")
def home():
    return "AI Skin Disease Detection API is running."


# ==========================================
# PREDICTION ROUTE
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty file"}), 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    disease, confidence = predict_disease(filepath)

    result = {
        "Detected Disease": disease,
        "Confidence (%)": round(confidence, 2),
        "Recommendation": "Consult a dermatologist for confirmation."
    }

    return jsonify(result)


# ==========================================
# RUN SERVER
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)
