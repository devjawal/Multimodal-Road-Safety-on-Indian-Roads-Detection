import os
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import shutil

# Configuration
NUM_SAMPLES_PER_CLASS = 100
IMG_SIZE = (224, 224)
DATA_DIR = "synthetic_data"
IMG_DIR = os.path.join(DATA_DIR, "images")

HAZARD_CLASSES = {
    "normal": {"color": (128, 128, 128), "text": ""},
    "pothole": {"color": (80, 50, 20), "text": "Pothole"},
    "sharp_turn": {"color": (255, 255, 0), "text": "Turn ->"},
    "object_on_road": {"color": (255, 0, 0), "text": "Object!"},
    "slippery_road": {"color": (0, 191, 255), "text": "Slippery"},
    "landslide": {"color": (139, 69, 19), "text": "Landslide"},
    "animal_crossing": {"color": (0, 255, 0), "text": "Animal"},
}


def generate_sensor_data(hazard_class):
    """
    Generates more distinct and realistic synthetic sensor data based on the hazard class.
    The goal is to create clearly separable data for the model to achieve high precision.
    """
    # Baseline: Normal driving conditions
    speed = np.random.uniform(60, 90)          # Highway cruising speed
    acceleration = np.random.normal(0, 0.2)   # Stable speed, minimal acceleration
    vibration = np.random.uniform(5, 10)       # Smooth road vibration
    temp = np.random.uniform(18, 35)     # Normal operating temperature

    if hazard_class == "pothole":
        # A sudden, sharp jolt.
        speed = np.random.uniform(30, 60)
        acceleration = np.random.normal(-2.5, 0.5) # Deceleration from the impact
        vibration = np.random.uniform(80, 120)     # Very high, sharp vibration spike
    
    elif hazard_class == "sharp_turn":
        # Slowing down for a curve.
        speed = np.random.uniform(25, 45)
        acceleration = np.random.normal(-1.5, 0.5) # Braking into the turn
        vibration = np.random.uniform(15, 25)      # Higher tire/suspension noise from turning
        
    elif hazard_class == "object_on_road":
        # Emergency braking for a static object.
        speed = np.random.uniform(0, 25)
        acceleration = np.random.normal(-4.5, 1.0) # Hard, sudden braking
        vibration = np.random.uniform(20, 35)      # Vibration from hard braking (ABS, etc.)
        
    elif hazard_class == "slippery_road":
        # Cautious driving on an unstable surface.
        speed = np.random.uniform(20, 50)
        acceleration = np.random.normal(0, 1.5)   # Unstable, erratic acceleration/deceleration
        vibration = np.random.uniform(25, 45)      # Vibration from tire slippage or rough ice
        temp = np.random.uniform(-10, 5)     # KEY INDICATOR: cold/icy conditions
        
    elif hazard_class == "landslide":
        # Emergency stop, significant environmental vibration.
        speed = np.random.uniform(0, 15)
        acceleration = np.random.normal(-5.0, 1.0) # Emergency stop
        vibration = np.random.uniform(50, 90)      # A sustained rumble, different from a pothole spike
        
    elif hazard_class == "animal_crossing":
        # Sudden but controlled braking for a moving object.
        speed = np.random.uniform(10, 40)
        acceleration = np.random.normal(-3.5, 0.8) # Hard but controlled braking
        vibration = np.random.uniform(15, 30)
        
    # The "normal" class uses the baseline values defined at the start.
    
    return round(speed, 2), round(acceleration, 2), round(vibration, 2), round(temp, 2)

def generate_image(filepath, hazard_info):
    """Creates a simple synthetic image representing a hazard."""
    img = Image.new('RGB', IMG_SIZE, color=(200, 200, 200))
    draw = ImageDraw.Draw(img)

    # Draw a "road"
    draw.rectangle([0, IMG_SIZE[1] * 0.6, IMG_SIZE[0], IMG_SIZE[1]], fill=(80, 80, 80))

    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except IOError:
        font = ImageFont.load_default()

    # Draw hazard representation
    if hazard_info["text"]:
        shape_color = hazard_info["color"]
        text = hazard_info["text"]
        draw.ellipse([80, 80, 160, 160], fill=shape_color, outline=(0,0,0))
        draw.text((85, 105), text, fill=(255, 255, 255), font=font)

    img.save(filepath)


def generate_data():
    """Main function to generate the entire dataset."""
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)

    os.makedirs(IMG_DIR, exist_ok=True)
    print(f"Generating synthetic data in '{DATA_DIR}' directory...")

    dataset = []
    for class_name, info in HAZARD_CLASSES.items():
        for i in range(NUM_SAMPLES_PER_CLASS):
            img_filename = f"{class_name}_{i+1}.png"
            filepath = os.path.join(IMG_DIR, img_filename)

            # Generate and save image
            generate_image(filepath, info)

            # Generate sensor data
            speed, acc, vib, temp = generate_sensor_data(class_name)

            dataset.append({
                "image_path": filepath,
                "speed_kmh": speed,
                "acceleration_ms2": acc,
                "vibration_hz": vib,
                "temperature_c": temp,
                "hazard": class_name
            })
    
    # Create and save DataFrame
    df = pd.DataFrame(dataset)
    csv_path = os.path.join(DATA_DIR, "sensor_data.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"Generated {len(df)} samples.")
    print(f"Image data saved in: {IMG_DIR}")
    print(f"Sensor data saved to: {csv_path}")


if __name__ == "__main__":
    generate_data()
