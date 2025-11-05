# file: predict.py
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- Configuration ---
IMG_SIZE = (224, 224)
SENSOR_FEATURES = ['speed_kmh', 'acceleration_ms2', 'vibration_hz', 'temperature_c']
NORMAL_CLASS_LABEL = 'normal'

# --- Load Saved Models and Encoder ---
print("Loading models and encoder...")
try:
    vision_model = load_model("vision_model.h5")
    sensor_model = joblib.load("sensor_model.pkl")
    fusion_model = load_model("fusion_model.h5")
    label_encoder = joblib.load("label_encoder.pkl")
    print("Models and encoder loaded successfully.")
except Exception as e:
    print(f"Error loading files: {e}")
    print("Please ensure you have run the training script to generate the model files.")
    exit()

def prepare_image(img_path):
    """Loads and preprocesses an image for the vision model."""
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

def get_prediction(image_path, sensor_data_dict):
    """
    Takes an image path and sensor data, and returns an alert decision.
    """
    # 1. Prepare inputs
    processed_image = prepare_image(image_path)
    sensor_data_df = pd.DataFrame([sensor_data_dict])

    # 2. Get individual model predictions
    vision_pred_proba = vision_model.predict(processed_image, verbose=0)
    sensor_pred_proba = sensor_model.predict_proba(sensor_data_df[SENSOR_FEATURES])

    # 3. Combine for fusion model
    fusion_input = np.concatenate([vision_pred_proba, sensor_pred_proba], axis=1)
    fusion_pred_proba = fusion_model.predict(fusion_input, verbose=0)

    # 4. Decode the final prediction
    predicted_class_index = np.argmax(fusion_pred_proba)
    predicted_hazard = label_encoder.classes_[predicted_class_index]
    confidence = fusion_pred_proba[0][predicted_class_index]

    print(f"\n--- Prediction Result ---")
    print(f"Model Prediction: '{predicted_hazard}'")
    print(f"Confidence: {confidence:.2%}")

    # 5. Make the final alert decision
    if predicted_hazard != NORMAL_CLASS_LABEL and confidence > 0.50:
        return f"🚨 ALERT! Hazard Detected: {predicted_hazard.upper()}"
    else:
        return f"✅ Status: Clear. No hazard detected."

if __name__ == '__main__':
    # --- EXAMPLE USAGE ---
    # Replace with the path to an image you want to test
    test_image_path = "real_world_data/sharp_turn/Image_7.jpg" 

    # Replace with the sensor data for that specific moment
    test_sensor_data = {
        "speed_kmh": 35.0,         # Lower speed for a turn
        "acceleration_ms2": -1.5,      # Braking
        "vibration_hz": 20.0,        # Moderate vibration, not a sharp jolt
        "temperature_c": 28.0
    }

    # Get the final decision
    decision = get_prediction(test_image_path, test_sensor_data)
    
    print("\n--- Final Decision ---")
    print(decision)