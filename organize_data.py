import os
import shutil
import random
import pandas as pd

# --- Configuration ---
SOURCE_DIR = "real_world_data"
TRAIN_DIR = os.path.join(SOURCE_DIR, "train")
TEST_DIR = os.path.join(SOURCE_DIR, "test")
CSV_PATH = os.path.join(SOURCE_DIR, "sensor_data_real.csv")
SPLIT_RATIO = 0.85 # 85% for training, 15% for testing

# Import sensor generation from our original script
from data_generator import generate_sensor_data

def organize_images_and_create_csv():
    """
    Splits downloaded images into train/test sets and generates a corresponding CSV
    with sensor data.
    """
    if os.path.exists(TRAIN_DIR):
        print("Train/Test directories already exist. Skipping organization.")
        if os.path.exists(CSV_PATH):
            return pd.read_csv(CSV_PATH)
    
    # Get class names from the folder names
    classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d)) and d not in ['train', 'test']]
    
    # Create train and test directories
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    all_data = []

    print("\n--- Organizing images into train/test sets ---")
    for hazard_class in classes:
        print(f"Processing class: {hazard_class}")
        
        # Create class subdirectories in train and test
        os.makedirs(os.path.join(TRAIN_DIR, hazard_class), exist_ok=True)
        os.makedirs(os.path.join(TEST_DIR, hazard_class), exist_ok=True)

        # Get all image files for the class
        class_path = os.path.join(SOURCE_DIR, hazard_class)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)

        # Split files
        split_index = int(len(images) * SPLIT_RATIO)
        train_files = images[:split_index]
        test_files = images[split_index:]

        # Move files and generate data
        for img_list, dest_dir in [(train_files, TRAIN_DIR), (test_files, TEST_DIR)]:
            for img_name in img_list:
                # Copy file
                src_path = os.path.join(class_path, img_name)
                dest_path = os.path.join(dest_dir, hazard_class, img_name)
                shutil.copy(src_path, dest_path)

                # Generate sensor data
                speed, acc, vib, temp = generate_sensor_data(hazard_class)
                
                # Append to our data list
                all_data.append({
                    "image_path": dest_path,
                    "speed_kmh": speed,
                    "acceleration_ms2": acc,
                    "vibration_hz": vib,
                    "temperature_c": temp,
                    "hazard": hazard_class,
                    "split": 'train' if dest_dir == TRAIN_DIR else 'test'
                })

    # Create and save the final DataFrame
    df = pd.DataFrame(all_data)
    df.to_csv(CSV_PATH, index=False)
    print(f"\nSuccessfully organized images and created CSV at: {CSV_PATH}")
    return df

if __name__ == "__main__":
    organize_images_and_create_csv()
