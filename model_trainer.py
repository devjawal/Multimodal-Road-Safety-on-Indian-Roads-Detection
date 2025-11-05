import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Concatenate, Dropout
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# --- Configuration ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SENSOR_FEATURES = ['speed_kmh', 'acceleration_ms2', 'vibration_hz', 'temperature_c']

def prepare_data(train_df, test_df):
    """Encodes labels for training."""
    label_encoder = LabelEncoder()
    
    # Fit on the combination of train and test to ensure all classes are captured
    all_labels = pd.concat([train_df['hazard'], test_df['hazard']]).unique()
    label_encoder.fit(all_labels)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Testing samples: {len(test_df)}")
    
    return label_encoder

def train_vision_model(train_df, num_classes):
    """Trains the MobileNetV2-based vision model with data augmentation."""
    print("\n--- Training Vision Model ---")
    
    # Data Augmentation for training images to improve model robustness
    train_datagen = ImageDataGenerator(
        rescale=1./255.,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2 # Use 20% of training data for validation
    )
    
    # No augmentation for validation data, just rescaling
    validation_datagen = ImageDataGenerator(rescale=1./255., validation_split=0.2)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='image_path',
        y_col='hazard',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = validation_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='image_path',
        y_col='hazard',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    base_model.trainable = False # Freeze base layers

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x) # Add dropout for regularization
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(
        train_generator,
        epochs=10, # Increased epochs for better learning on real data
        validation_data=validation_generator
    )
    

    model.save("vision_model.h5")
    print("Vision model training complete. Model saved to 'vision_model.h5'")
    
    return model

def train_sensor_model(train_df, test_df):
    """Trains the RandomForest model on sensor data."""
    print("\n--- Training Sensor Model ---")
    
    le_sensor = LabelEncoder()
    y_train_encoded = le_sensor.fit_transform(train_df['hazard'])
    y_test_encoded = le_sensor.transform(test_df['hazard'])
    
    X_train = train_df[SENSOR_FEATURES]
    X_test = test_df[SENSOR_FEATURES]
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train_encoded)
    
    
    y_pred = model.predict(X_test)
    print("Sensor Model Performance:")
    print(f"Accuracy: {accuracy_score(y_test_encoded, y_pred):.4f}")
    
    joblib.dump(model, 'sensor_model.pkl')
    print("Sensor model training complete. Model saved to 'sensor_model.pkl'")
    return model


def train_fusion_model(vision_model, sensor_model, train_df, test_df, label_encoder):
    """Trains a neural network to fuse predictions from both models."""
    print("\n--- Training Fusion Model ---")
    num_classes = len(label_encoder.classes_)
    
    train_df_encoded = train_df.copy()
    train_df_encoded['hazard_encoded'] = label_encoder.transform(train_df_encoded['hazard'])

    train_img_gen = ImageDataGenerator(rescale=1./255.).flow_from_dataframe(
        dataframe=train_df_encoded, x_col='image_path', class_mode=None, target_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False)
    vision_preds_train = vision_model.predict(train_img_gen)

    sensor_preds_train = sensor_model.predict_proba(train_df_encoded[SENSOR_FEATURES])
    
    fusion_X_train = np.concatenate([vision_preds_train, sensor_preds_train], axis=1)
    fusion_y_train = tf.keras.utils.to_categorical(train_df_encoded['hazard_encoded'], num_classes=num_classes)
    
    input_shape = (fusion_X_train.shape[1],)
    fusion_input = Input(shape=input_shape)
    x = Dense(64, activation='relu')(fusion_input)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu')(x)
    fusion_output = Dense(num_classes, activation='softmax')(x)
    
    fusion_model = Model(inputs=fusion_input, outputs=fusion_output)
    fusion_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    fusion_model.fit(fusion_X_train, fusion_y_train, epochs=20, batch_size=BATCH_SIZE, validation_split=0.2, verbose=1)
    
    fusion_model.save("fusion_model.h5")
    print("Fusion model training complete. Model saved to 'fusion_model.h5'")
    return fusion_model
    
def evaluate_model(model, test_df, label_encoder, model_type='fusion', vision_model=None, sensor_model=None):
    """Evaluates the model and prints a classification report and confusion matrix."""
    print(f"\n--- Evaluating {model_type.title()} Model ---")
    
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    test_df_encoded = test_df.copy()
    test_df_encoded['hazard_encoded'] = label_encoder.transform(test_df_encoded['hazard'])
    y_true = test_df_encoded['hazard_encoded']
    
    if model_type == 'vision':
        test_img_gen = ImageDataGenerator(rescale=1./255.).flow_from_dataframe(
            dataframe=test_df_encoded, x_col='image_path', class_mode=None, target_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False)
        y_pred_proba = model.predict(test_img_gen)
        y_pred = np.argmax(y_pred_proba, axis=1)
    elif model_type == 'sensor':
        y_pred = model.predict(test_df_encoded[SENSOR_FEATURES])
    elif model_type == 'fusion':
        test_img_gen = ImageDataGenerator(rescale=1./255.).flow_from_dataframe(
            dataframe=test_df_encoded, x_col='image_path', class_mode=None, target_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False)
        vision_preds_test = vision_model.predict(test_img_gen)
        sensor_preds_test = sensor_model.predict_proba(test_df_encoded[SENSOR_FEATURES])
        
        fusion_X_test = np.concatenate([vision_preds_test, sensor_preds_test], axis=1)
        y_pred_proba = model.predict(fusion_X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
    
    print(f"Classification Report for {model_type.title()} Model:")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(label_encoder.classes_)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f'{model_type.title()} Model Confusion Matrix (Real Data)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(output_dir, f'{model_type}_confusion_matrix_real.png'))
    # plt.show() # Commented out to prevent blocking script execution
    plt.close()

