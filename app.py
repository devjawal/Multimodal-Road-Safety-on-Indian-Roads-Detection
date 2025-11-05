import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

IMG_SIZE = (224, 224)
SENSOR_FEATURES = ['speed_kmh', 'acceleration_ms2', 'vibration_hz', 'temperature_c']
NORMAL_CLASS_LABEL = 'normal'
CONFIDENCE_THRESHOLD = 0.50

# --- Load Saved Models and Encoder ONCE at Startup ---
# This is crucial for performance. Models are loaded into memory when the server starts.
print("Loading pre-trained models and encoder...")
try:
    vision_model = load_model("vision_model.h5")
    sensor_model = joblib.load("sensor_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    print("Models and encoder loaded successfully.")
except Exception as e:
    print(f"--- FATAL ERROR ---")
    print(f"Error loading model files: {e}")
    print("Please ensure 'vision_model.h5', 'sensor_model.pkl', and 'label_encoder.pkl' are in the same directory as app.py")
    # In a real app, you might exit or handle this more gracefully.

def prepare_image(img_path):
    """Loads and preprocesses an image for the vision model."""
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

def get_prediction(image_path, sensor_data_dict):
    """
    Takes an image path and sensor data, and returns a detailed alert decision.
    """
    # 1. Prepare inputs
    processed_image = prepare_image(image_path)
    sensor_data_df = pd.DataFrame([sensor_data_dict])

    # 2. Get individual model predictions (using the globally loaded models)
    vision_pred_proba = vision_model.predict(processed_image, verbose=0)[0]
    vision_class_index = np.argmax(vision_pred_proba)
    vision_hazard = label_encoder.classes_[vision_class_index]
    vision_confidence = vision_pred_proba[vision_class_index]

    sensor_pred_proba = sensor_model.predict_proba(sensor_data_df[SENSOR_FEATURES])[0]
    sensor_class_index = np.argmax(sensor_pred_proba)
    sensor_hazard = label_encoder.classes_[sensor_class_index]
    sensor_confidence = sensor_pred_proba[sensor_class_index]

    # Create a dictionary of individual model outputs to pass to the template
    details = {
        "vision_hazard": vision_hazard,
        "vision_confidence": f"{vision_confidence:.2%}",
        "sensor_hazard": sensor_hazard,
        "sensor_confidence": f"{sensor_confidence:.2%}"
    }

    # 3. Custom Fusion Logic
    vision_is_hazard = vision_hazard != NORMAL_CLASS_LABEL and vision_confidence > CONFIDENCE_THRESHOLD
    sensor_is_hazard = sensor_hazard != NORMAL_CLASS_LABEL and sensor_confidence > CONFIDENCE_THRESHOLD

    if vision_is_hazard and sensor_is_hazard and vision_hazard == sensor_hazard:
        decision = f"ALERT! Hazard Confirmed: {vision_hazard.upper()}"
    elif vision_is_hazard and sensor_is_hazard and vision_hazard != sensor_hazard:
        decision = f"ALERT! Mixed Signals: VISION saw '{vision_hazard.upper()}' + SENSORS detected '{sensor_hazard.upper()}'"
    elif vision_is_hazard and not sensor_is_hazard:
        decision = f"ALERT! Vision System Detected: {vision_hazard.upper()}"
    elif not vision_is_hazard and sensor_is_hazard:
        decision = f"ALERT! Sensor System Detected: {sensor_hazard.upper()}"
    else:
        decision = "Status: Clear. No hazard detected."
        
    return decision, details

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if an image file was uploaded
        if 'image' not in request.files or request.files['image'].filename == '':
            return render_template('index.html', error="No image file selected.")
        
        image_file = request.files['image']
        filename = secure_filename(image_file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(image_path)

        # Get sensor data from the form, with type conversion
        try:
            sensor_data = {
                "speed_kmh": request.form.get('speed', type=float),
                "acceleration_ms2": request.form.get('acceleration', type=float),
                "vibration_hz": request.form.get('vibration', type=float),
                "temperature_c": request.form.get('temperature', type=float)
            }
        except (TypeError, ValueError):
            return render_template('index.html', error="Invalid sensor data. Please enter numbers only.")

        # Get the prediction and detailed outputs
        decision, details = get_prediction(image_path, sensor_data)
        
        # Render the page again with the results
        return render_template('index.html', result=decision, details=details, image_file=filename)

    # Initial page load (GET request)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
