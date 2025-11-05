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
CONFIDENCE_THRESHOLD = 0.50 # Only consider predictions above 50% confidence

# --- Load Saved Models and Encoder ---
print("Loading models and encoder...")
try:
    vision_model = load_model("vision_model.h5")
    sensor_model = joblib.load("sensor_model.pkl")
    # The fusion_model is no longer needed for this logic, but we load the encoder
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
    Takes an image path and sensor data, and returns a detailed alert decision
    based on custom fusion logic.
    """
    # 1. Prepare inputs
    processed_image = prepare_image(image_path)
    sensor_data_df = pd.DataFrame([sensor_data_dict])

    # 2. Get VISION model's prediction and confidence
    vision_pred_proba = vision_model.predict(processed_image, verbose=0)[0]
    vision_class_index = np.argmax(vision_pred_proba)
    vision_hazard = label_encoder.classes_[vision_class_index]
    vision_confidence = vision_pred_proba[vision_class_index]

    # 3. Get SENSOR model's prediction and confidence
    sensor_pred_proba = sensor_model.predict_proba(sensor_data_df[SENSOR_FEATURES])[0]
    sensor_class_index = np.argmax(sensor_pred_proba)
    sensor_hazard = label_encoder.classes_[sensor_class_index]
    sensor_confidence = sensor_pred_proba[sensor_class_index]

    print("\n--- Individual Model Outputs ---")
    print(f"Vision Model saw: '{vision_hazard}' (Confidence: {vision_confidence:.2%})")
    print(f"Sensor Model felt: '{sensor_hazard}' (Confidence: {sensor_confidence:.2%})")

    # 4. Custom Fusion Logic to Determine Final Alert
    vision_is_hazard = vision_hazard != NORMAL_CLASS_LABEL and vision_confidence > CONFIDENCE_THRESHOLD
    sensor_is_hazard = sensor_hazard != NORMAL_CLASS_LABEL and sensor_confidence > CONFIDENCE_THRESHOLD

    # Case 1: Both models agree on the SAME hazard
    if vision_is_hazard and sensor_is_hazard and vision_hazard == sensor_hazard:
        return f"🚨 ALERT! Hazard Confirmed: {vision_hazard.upper()}"

    # Case 2: Both detect different hazards (your requested scenario)
    elif vision_is_hazard and sensor_is_hazard and vision_hazard != sensor_hazard:
        return f"🚨 ALERT! Mixed Signals: VISION saw '{vision_hazard.upper()}' + SENSORS detected '{sensor_hazard.upper()}'"

    # Case 3: Only the vision model detects a hazard
    elif vision_is_hazard and not sensor_is_hazard:
        return f"🚨 ALERT! Vision System Detected: {vision_hazard.upper()}"

    # Case 4: Only the sensor model detects a hazard
    elif not vision_is_hazard and sensor_is_hazard:
        return f"🚨 ALERT! Sensor System Detected: {sensor_hazard.upper()}"

    # Case 5: Neither model detects a hazard with enough confidence
    else:
        return f"✅ Status: Clear. No hazard detected."


if __name__ == '__main__':
    # --- EXAMPLE USAGE ---
    # Replace with the path to an image you want to test
    test_image_path = "real_world_data/slippery_road/Image_10.jpg" 

    # Replace with sensor data that might indicate a DIFFERENT hazard (e.g., pothole)
    test_sensor_data = {
        "speed_kmh": 85.5,
        "acceleration_ms2": -1.1,
        "vibration_hz": 45.8, # High vibration, suggests a pothole
        "temperature_c": 28.0
    }

    # Get the final decision
    decision = get_prediction(test_image_path, test_sensor_data)
    
    print("\n--- Final Decision ---")
    print(decision)