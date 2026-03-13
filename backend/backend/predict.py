# ==========================================
# AI SKIN DISEASE DETECTION - PREDICTION
# ==========================================

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = tf.keras.models.load_model("../model/skin_model.h5")

# ==========================================
# CLASS LABELS
# (Must match dataset folder names)
# ==========================================

class_labels = [
    "acne",
    "eczema",
    "melanoma",
    "psoriasis"
]

# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_disease(img_path):

    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    disease = class_labels[predicted_index]

    return disease, confidence


# ==========================================
# TEST PREDICTION
# ==========================================

if __name__ == "__main__":

    image_path = input("Enter skin image path: ")

    disease, confidence = predict_disease(image_path)

    print("\n===== AI DIAGNOSIS =====")
    print("Detected Disease :", disease)
    print("Confidence :", round(confidence,2), "%")
    print("Recommendation : Consult a dermatologist")
    print("========================")
