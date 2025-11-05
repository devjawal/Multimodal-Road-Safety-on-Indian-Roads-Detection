import pandas as pd
import numpy as np
import json
from datetime import datetime
import time
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Configuration
LOG_FILE = 'hazard_alerts.jsonl'
IMG_SIZE = (224, 224)
SENSOR_FEATURES = ['speed_kmh', 'acceleration_ms2', 'vibration_hz', 'temperature_c']

def prepare_image(img_path):
    """Loads and preprocesses an image for the vision model."""
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

def simulate_realtime_detection(test_df, vision_model, sensor_model, fusion_model, label_encoder):
    """
    Simulates real-time hazard detection on the test set and logs alerts.
    """
    print("\n--- Starting Real-Time Hazard Detection Simulation ---")
    
    # Clean previous log file
    with open(LOG_FILE, 'w') as f:
        pass # Create an empty file

    for index, row in test_df.iterrows():
        print(f"\nProcessing sample {index}...")
        
        # 1. Get Vision Model Prediction
        image_path = row['image_path']
        processed_image = prepare_image(image_path)
        vision_pred_proba = vision_model.predict(processed_image, verbose=0)

        # 2. Get Sensor Model Prediction
        sensor_data = pd.DataFrame([row[SENSOR_FEATURES]])
        sensor_pred_proba = sensor_model.predict_proba(sensor_data)
        
        # 3. Get Fusion Model Prediction
        fusion_input = np.concatenate([vision_pred_proba, sensor_pred_proba], axis=1)
        fusion_pred_proba = fusion_model.predict(fusion_input, verbose=0)
        
        # Determine final prediction
        predicted_class_index = np.argmax(fusion_pred_proba)
        predicted_hazard = label_encoder.classes_[predicted_class_index]
        confidence = fusion_pred_proba[0][predicted_class_index]

        print(f"  -> True Hazard: {row['hazard']}")
        print(f"  -> Predicted Hazard: {predicted_hazard} (Confidence: {confidence:.2f})")

        # 4. Log Alert only if a hazard is detected with high confidence
        if predicted_hazard != 'normal' and confidence > 0.49:
            alert = {
                "timestamp": datetime.now().isoformat(),
                "predicted_hazard": predicted_hazard,
                "confidence": float(confidence),
                "speed_kmh": row['speed_kmh'],
                "acceleration_ms2": row['acceleration_ms2'],
                "image_path": image_path
            }
            with open(LOG_FILE, 'a') as f:
                f.write(json.dumps(alert) + '\n')
            
            print(f"  [!] ALERT LOGGED: {predicted_hazard}")
        else:
            # Add a message for skipped alerts for clarity
            if predicted_hazard != 'normal':
                print(f"  -> Skipping alert for '{predicted_hazard}' due to low confidence.")

        # Simulate real-time delay
        time.sleep(1)
        
    print(f"\nSimulation finished. Alerts logged to '{LOG_FILE}'.")