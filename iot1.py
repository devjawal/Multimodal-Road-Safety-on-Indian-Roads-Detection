#!/usr/bin/env python3
"""
road_safety_software_only.py

Software-only Road Hazard Detection with ML + Deep Learning.

- Generates a synthetic dataset (images + sensor features) or can be adapted to real datasets.
- Trains:
    * Vision embedding + classifier (MobileNetV2 transfer learning)
    * Sensor classifier (RandomForest)
    * Fusion model (Keras MLP) that concatenates vision embedding + sensor features
- Runs inference on an input JSON stream to produce per-event hazard probabilities and writes alerts.

Author: ChatGPT-based template for your project (adapt & extend as needed).
"""
import os
import json
import math
import random
import argparse
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras import layers, models, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical

# scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt

# ------------------------
# Config
# ------------------------
IMG_SIZE = (128, 128)  # small for quick training
NUM_SYNTH_IMAGES = 600  # overall synthetic images
CLASSES = ["none", "pothole", "speed_breaker", "animal", "stalled_vehicle"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}
VISION_EMB_SIZE = 128  # size of embedding extracted from vision model
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------------
# Synthetic image generator (simple shapes to represent hazards)
# ------------------------
def draw_synthetic_hazard(hazard: str, img_size=IMG_SIZE) -> Image.Image:
    """
    Create a simple synthetic "road-like" image and overlay a shape representing hazard.
    - pothole: dark circle
    - speed_breaker: bright elongated ellipse
    - animal: small blob (brown)
    - stalled_vehicle: rectangle
    - none: plain road
    """
    w, h = img_size
    # base road background
    base = Image.new("RGB", (w, h), (60, 60, 60))
    d = ImageDraw.Draw(base)
    # draw lane center
    for y in range(0, h, 12):
        d.rectangle([w//2 - 3, y, w//2 + 3, y+6], fill=(200,200,200))
    # overlay hazard shapes in lower half
    cx = random.randint(w//3, 2*w//3)
    cy = random.randint(h//2+10, h-20)
    if hazard == "pothole":
        r = random.randint(8, 20)
        d.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(20, 20, 20))
    elif hazard == "speed_breaker":
        rx = random.randint(30, 60)
        ry = random.randint(6, 14)
        d.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=(210,210,210))
    elif hazard == "animal":
        # small brown blob with some legs
        r = random.randint(8, 16)
        d.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(120,70,30))
        # legs
        d.rectangle([cx-6, cy+r-2, cx-4, cy+r+6], fill=(50,30,20))
    elif hazard == "stalled_vehicle":
        # rectangle car
        rw = random.randint(30, 60)
        rh = random.randint(12, 22)
        d.rectangle([cx-rw//2, cy-rh, cx+rw//2, cy+rh], fill=(30,30,120))
        # wheels
        d.ellipse([cx-rw//2-6, cy+rh-3, cx-rw//2+6, cy+rh+9], fill=(20,20,20))
        d.ellipse([cx+rw//2-6, cy+rh-3, cx+rw//2+6, cy+rh+9], fill=(20,20,20))
    # add random brightness variation
    return base

# ------------------------
# Sensor data generator
# ------------------------
def synth_sensor_for_hazard(hazard: str) -> Dict:
    """
    Return sensor dict: speed_kmh, accel_z_g, rain_prob, light_level
    Heuristics:
      - pothole: accel spike moderate
      - speed_breaker: small accel bump
      - animal: speed lower, possibly braking pattern (accel negative/low)
      - stalled_vehicle: speed near 0
      - none: normal driving
    """
    if hazard == "pothole":
        return {
            "speed_kmh": float(max(10, random.gauss(40, 8))),
            "accel_z_g": float(max(0.1, random.gauss(0.9, 0.5))),
            "rain_prob": float(max(0.0, min(1.0, random.gauss(0.1, 0.15)))),
            "light_level": random.choice(["day","day","evening"])
        }
    if hazard == "speed_breaker":
        return {
            "speed_kmh": float(max(5, random.gauss(30, 6))),
            "accel_z_g": float(max(0.05, random.gauss(0.35, 0.2))),
            "rain_prob": float(max(0.0, min(1.0, random.gauss(0.05, 0.1)))),
            "light_level": random.choice(["day","day","evening"])
        }
    if hazard == "animal":
        return {
            "speed_kmh": float(max(0, random.gauss(18, 6))),
            "accel_z_g": float(max(0.0, random.gauss(0.25, 0.3))),  # may show braking patterns
            "rain_prob": float(max(0.0, min(1.0, random.gauss(0.05, 0.15)))),
            "light_level": random.choice(["day","evening","night"])
        }
    if hazard == "stalled_vehicle":
        return {
            "speed_kmh": float(max(0.0, random.gauss(4, 3))),
            "accel_z_g": float(max(0.0, random.gauss(0.05, 0.05))),
            "rain_prob": float(max(0.0, min(1.0, random.gauss(0.05, 0.1)))),
            "light_level": random.choice(["day","evening"])
        }
    # none
    return {
        "speed_kmh": float(max(5, random.gauss(45, 10))),
        "accel_z_g": float(max(0.0, random.gauss(0.15, 0.12))),
        "rain_prob": float(max(0.0, min(1.0, random.gauss(0.05, 0.07)))),
        "light_level": random.choice(["day","day","evening"])
    }

# ------------------------
# Prepare synthetic dataset on disk (images + sensor features + labels)
# ------------------------
def build_synthetic_dataset(num_images: int = NUM_SYNTH_IMAGES, out_dir: str = "synthetic_data"):
    os.makedirs(out_dir, exist_ok=True)
    images = []
    sensors = []
    labels = []
    # balances classes roughly
    for i in range(num_images):
        # sample label with some distribution
        label = random.choices(CLASSES, weights=[0.4,0.2,0.15,0.15,0.1])[0]
        img = draw_synthetic_hazard(label)
        fname = f"{label}_{i:04d}.png"
        path = os.path.join(out_dir, fname)
        img.save(path)
        sensor = synth_sensor_for_hazard(label)
        images.append(path)
        sensors.append(sensor)
        labels.append(CLASS_TO_IDX[label])
    # Save sensor CSV/JSON
    meta = []
    for p, s, l in zip(images, sensors, labels):
        meta.append({
            "image": p,
            "label": int(l),
            "speed_kmh": float(s["speed_kmh"]),
            "accel_z_g": float(s["accel_z_g"]),
            "rain_prob": float(s["rain_prob"]),
            "light_level": s["light_level"]
        })
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Built synthetic dataset at {out_dir} with {len(images)} samples")
    return meta

# ------------------------
# Vision model (transfer learning) -> produce embedding of size VISION_EMB_SIZE
# ------------------------
def build_vision_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), embed_size=VISION_EMB_SIZE, trainable_layers=20):
    base = MobileNetV2(input_shape=input_shape, include_top=False, pooling="avg", weights="imagenet")
    # optionally freeze most layers
    for layer in base.layers[:-trainable_layers]:
        layer.trainable = False
    x = base.output
    x = layers.Dense(embed_size, activation="relu", name="vision_emb")(x)
    # classification head (optional during training)
    out = layers.Dense(len(CLASSES), activation="softmax", name="vision_out")(x)
    model = Model(inputs=base.input, outputs=[out, x])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss={"vision_out":"categorical_crossentropy"},
                  metrics={"vision_out":"accuracy"})
    return model

# ------------------------
# Sensor feature vector builder
# ------------------------
def sensor_to_vector(sensor_dict):
    # numeric features: speed_kmh, accel_z_g, rain_prob
    # categorical light_level -> one-hot
    light = sensor_dict.get("light_level","day")
    light_onehot = [
        1.0 if light == "day" else 0.0,
        1.0 if light == "evening" else 0.0,
        1.0 if light == "night" else 0.0
    ]
    return np.array([
        float(sensor_dict.get("speed_kmh",0.0)),
        float(sensor_dict.get("accel_z_g",0.0)),
        float(sensor_dict.get("rain_prob",0.0))
    ] + light_onehot, dtype=np.float32)

# ------------------------
# Load images and sensor features for training
# ------------------------
def load_dataset_from_meta(meta_list):
    X_imgs = []
    X_sensors = []
    y = []
    for m in meta_list:
        img_path = m["image"]
        img = Image.open(img_path).resize(IMG_SIZE)
        arr = img_to_array(img) / 255.0
        X_imgs.append(arr)
        sensor = {"speed_kmh":m["speed_kmh"], "accel_z_g":m["accel_z_g"],
                  "rain_prob":m["rain_prob"], "light_level":m.get("light_level","day")}
        X_sensors.append(sensor_to_vector(sensor))
        y.append(int(m["label"]))
    return np.array(X_imgs, dtype=np.float32), np.array(X_sensors, dtype=np.float32), np.array(y, dtype=np.int32)

# ------------------------
# Train sensor ML model (RandomForest)
# ------------------------
def train_sensor_model(X_sensors, y, save_path=os.path.join(MODEL_DIR,"sensor_rf.joblib")):
    rf = RandomForestClassifier(n_estimators=100, random_state=SEED)
    rf.fit(X_sensors, y)
    joblib.dump(rf, save_path)
    print(f"Trained RandomForest sensor model -> {save_path}")
    return rf

# ------------------------
# Extract vision embeddings (train vision model first, then extract embedding layer output)
# ------------------------
def train_vision_model(X_imgs, y, save_path=os.path.join(MODEL_DIR,"vision_model.h5"), epochs=8, batch_size=16):
    y_cat = to_categorical(y, num_classes=len(CLASSES))
    model = build_vision_model()
    print(model.summary())
    # train classification head (also learns embedding)
    model.fit(X_imgs, {"vision_out": y_cat}, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=2)
    model.save(save_path)
    print(f"Saved vision model to {save_path}")
    return model

def extract_embeddings_from_vision(model, X_imgs):
    # model outputs [vision_out, embedding]
    preds, embs = model.predict(X_imgs, verbose=0)
    return embs  # shape (N, VISION_EMB_SIZE)

# ------------------------
# Fusion model (concatenate emb + sensor features)
# ------------------------
def build_and_train_fusion(emb_train, sensor_train, y_train,
                           emb_val, sensor_val, y_val,
                           save_path=os.path.join(MODEL_DIR,"fusion_model.h5"),
                           epochs=20, batch_size=32):
    # inputs
    inp_emb = layers.Input(shape=(emb_train.shape[1],), name="emb_inp")
    inp_sensor = layers.Input(shape=(sensor_train.shape[1],), name="sensor_inp")
    x = layers.Concatenate()([inp_emb, inp_sensor])
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    out = layers.Dense(len(CLASSES), activation="softmax")(x)
    model = Model(inputs=[inp_emb, inp_sensor], outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    y_train_cat = to_categorical(y_train, num_classes=len(CLASSES))
    y_val_cat = to_categorical(y_val, num_classes=len(CLASSES))
    model.fit([emb_train, sensor_train], y_train_cat,
              validation_data=([emb_val, sensor_val], y_val_cat),
              epochs=epochs, batch_size=batch_size, verbose=2)
    model.save(save_path)
    print(f"Saved fusion model -> {save_path}")
    return model

# ------------------------
# Evaluation helper
# ------------------------
def evaluate_models(vision_model, rf_model, fusion_model, X_imgs, X_sensors, y_true):
    # vision preds
    preds_v, _ = vision_model.predict(X_imgs, verbose=0)
    y_v = np.argmax(preds_v, axis=1)

    # sensor rf preds
    y_s = rf_model.predict(X_sensors)

    # emb for fusion
    _, embs = vision_model.predict(X_imgs, verbose=0)
    preds_f = fusion_model.predict([embs, X_sensors], verbose=0)
    y_f = np.argmax(preds_f, axis=1)

    print("=== Vision model classification report ===")
    print(classification_report(y_true, y_v, target_names=CLASSES))
    print("=== Sensor RF classification report ===")
    print(classification_report(y_true, y_s, target_names=CLASSES))
    print("=== Fusion model classification report ===")
    print(classification_report(y_true, y_f, target_names=CLASSES))

# ------------------------
# Inference pipeline (reads input.json events or synthetic stream)
# ------------------------
def inference_on_stream(fusion_model, vision_model, rf_model, input_events: List[Dict], alert_threshold=0.65):
    out_path = "hazard_events.jsonl"
    for ev in input_events:
        # get image (path or array or synthetic)
        img_path = ev.get("image", None)
        if img_path and os.path.exists(img_path):
            img = Image.open(img_path).resize(IMG_SIZE)
            arr = img_to_array(img) / 255.0
        else:
            # synthetic generation for event
            hazard_hint = ev.get("camera", {}).get("object", "none")
            img = draw_synthetic_hazard(hazard_hint, img_size=IMG_SIZE)
            arr = img_to_array(img) / 255.0

        arr = np.expand_dims(arr.astype(np.float32), axis=0)
        # vision embedding
        vision_outs = vision_model.predict(arr, verbose=0)
        # vision_model outputs ([vision_out], embedding)
        if isinstance(vision_outs, list) and len(vision_outs) == 2:
            # preds, emb
            preds_vis, emb = vision_outs
        else:
            preds_vis = vision_outs
            emb = np.zeros((1, VISION_EMB_SIZE), dtype=np.float32)

        # sensor vector
        sensor_raw = ev.get("sensors", {})
        sensor_vec = sensor_to_vector(sensor_raw).reshape(1, -1)

        # fusion predict
        fusion_probs = fusion_model.predict([emb, sensor_vec], verbose=0)[0]
        pred_idx = int(np.argmax(fusion_probs))
        pred_label = IDX_TO_CLASS[pred_idx]
        pred_prob = float(fusion_probs[pred_idx])

        ts = sensor_raw.get("timestamp", ev.get("timestamp", datetime.utcnow().isoformat()+"Z"))
        log = {
            "timestamp": ts,
            "predicted_label": pred_label,
            "predicted_prob": round(pred_prob, 4),
            "all_probs": {IDX_TO_CLASS[i]: float(round(float(p),4)) for i,p in enumerate(fusion_probs)},
            "sensor": sensor_raw
        }
        print(f"[{ts}] Predicted={pred_label} ({pred_prob:.3f}) probs={log['all_probs']}")

        if pred_prob >= alert_threshold and pred_label != "none":
            print("[ALERT] writing to", out_path)
            with open(out_path, "a") as f:
                f.write(json.dumps(log) + "\n")

# ------------------------
# Main: orchestrate generation -> train -> inference demo
# ------------------------
def main(args):
    # 1) build synthetic dataset (or load if exists)
    data_dir = "synthetic_data"
    meta_path = os.path.join(data_dir, "meta.json")
    if not os.path.exists(meta_path) or args.force_rebuild:
        meta = build_synthetic_dataset(num_images=args.num_images, out_dir=data_dir)
    else:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        print(f"Loaded existing metadata with {len(meta)} samples")

    # 2) load arrays
    X_imgs, X_sensors, y = load_dataset_from_meta(meta)
    # normalize sensor features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_sensors = scaler.fit_transform(X_sensors)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "sensor_scaler.joblib"))

    # 3) train/test split
    X_img_train, X_img_test, X_sens_train, X_sens_test, y_train, y_test = train_test_split(
        X_imgs, X_sensors, y, test_size=0.15, random_state=SEED, stratify=y)

    # 4) train vision model
    vision_model_path = os.path.join(MODEL_DIR, "vision_model.h5")
    vision_model = train_vision_model(X_img_train, y_train, save_path=vision_model_path, epochs=args.vision_epochs)

    # 5) extract embeddings
    emb_train = extract_embeddings_from_vision(vision_model, X_img_train)
    emb_test = extract_embeddings_from_vision(vision_model, X_img_test)

    # 6) train sensor RF
    sensor_rf = train_sensor_model(X_sens_train, y_train, save_path=os.path.join(MODEL_DIR,"sensor_rf.joblib"))

    # 7) train fusion model
    fusion_model = build_and_train_fusion(emb_train, X_sens_train, y_train, emb_test, X_sens_test, y_test,
                                         save_path=os.path.join(MODEL_DIR,"fusion_model.h5"),
                                         epochs=args.fusion_epochs)

    # 8) evaluate
    evaluate_models(vision_model, sensor_rf, fusion_model, X_img_test, X_sens_test, y_test)

    # 9) demo inference on sample input stream (from meta or synthetic)
    print("\n--- Running inference demo on 10 sample events ---")
    # prepare 10 sample events from meta
    demo_events = []
    for i in range(10):
        idx = random.randint(0, len(meta)-1)
        m = meta[idx]
        ev = {
            "image": m["image"],
            "timestamp": f"2025-08-26T10:{i:02d}:00Z",
            "sensors": {
                "speed_kmh": m["speed_kmh"],
                "accel_z_g": m["accel_z_g"],
                "rain_prob": m["rain_prob"],
                "light_level": m.get("light_level","day"),
                "timestamp": f"2025-08-26T10:{i:02d}:00Z"
            }
        }
        demo_events.append(ev)

    # load saved scaler and models from disk to mimic deployment
    loaded_scaler = joblib.load(os.path.join(MODEL_DIR, "sensor_scaler.joblib"))
    # scale demo sensors
    for ev in demo_events:
        vec = sensor_to_vector(ev["sensors"]).reshape(1,-1)
        sc = loaded_scaler.transform(vec).flatten()
        # update numeric fields to scaled values for inference function (fusion expects scaled)
        # We'll store scaled vector inside ev["sensors_scaled"] (fusion uses global scaler later)
        ev["sensors_scaled_vector"] = sc.tolist()

    # For inference we will design the code to scale using the same scaler
    # But simpler: reuse fusion_model and vision_model directly
    inference_on_stream(fusion_model, vision_model, sensor_rf, demo_events, alert_threshold=args.alert_threshold)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("road_safety_software_only.py")
    parser.add_argument("--num-images", type=int, default=NUM_SYNTH_IMAGES, help="Number of synthetic images to generate")
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild synthetic dataset even if exists")
    parser.add_argument("--vision-epochs", type=int, default=6, help="Epochs to train vision model (small for demo)")
    parser.add_argument("--fusion-epochs", type=int, default=12, help="Epochs to train fusion model")
    parser.add_argument("--alert-threshold", type=float, default=0.65, help="Threshold to trigger alerts")
    args = parser.parse_args()
    main(args)

