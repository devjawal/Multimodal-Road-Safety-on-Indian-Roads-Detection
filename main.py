import download_image
import organize_data
import model_trainer
import realtime_simulation
import pandas as pd
import os
import joblib

# --- Main Pipeline ---
def main():
    """Executes the full pipeline with real-world data."""
    
    # Step 1: Download Real-World Images if they don't exist
    print("--- Step 1: Checking for real-world image data ---")
    # This is the NEW, correct line
    download_image.main()
    # Step 2: Organize images and create the corresponding sensor data CSV
    print("\n--- Step 2: Organizing images and generating sensor data ---")
    full_df = organize_data.organize_images_and_create_csv()

    # Step 3: Get the train and test dataframes from the CSV
    train_df = full_df[full_df['split'] == 'train'].copy()
    test_df = full_df[full_df['split'] == 'test'].copy()
    
    # Step 4: Prepare Data for Training (Encode labels)
    print("\n--- Step 4: Preparing data for model training ---")
    label_encoder = model_trainer.prepare_data(train_df, test_df)

    joblib.dump(label_encoder, 'label_encoder.pkl')
    print("Label encoder saved to 'label_encoder.pkl'")
    
    num_classes = len(label_encoder.classes_)

    # Step 5: Train Models
    print("\n--- Step 5: Training all models ---")
    vision_model = model_trainer.train_vision_model(train_df, num_classes)
    sensor_model = model_trainer.train_sensor_model(train_df, test_df)
    fusion_model = model_trainer.train_fusion_model(vision_model, sensor_model, train_df, test_df, label_encoder)

    # Step 6: Evaluate All Models
    print("\n--- Step 6: Evaluating model performance ---")
    model_trainer.evaluate_model(vision_model, test_df, label_encoder, model_type='vision')
    model_trainer.evaluate_model(sensor_model, test_df, label_encoder, model_type='sensor')
    model_trainer.evaluate_model(
        fusion_model, 
        test_df, 
        label_encoder, 
        model_type='fusion',
        vision_model=vision_model,
        sensor_model=sensor_model
    )
    
    # Step 7: Run Real-Time Simulation
    print("\n--- Step 7: Starting real-time simulation ---")
    realtime_simulation.simulate_realtime_detection(test_df, vision_model, sensor_model, fusion_model, label_encoder)
    
    print("\n--- Project Pipeline Finished ---")

if __name__ == "__main__":
    main()

