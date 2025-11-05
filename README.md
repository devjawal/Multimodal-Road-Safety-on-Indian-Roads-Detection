# Road Safety on Indian Roads: An AI-Enabled Sensor Fusion System

This project is a real-time road hazard detection system developed at Vellore Institute of Technology (VIT). It uses a multimodal AI approach, fusing camera (vision) data with IoT sensor data to improve road safety on Indian roads.

## 🧑‍💻 Project Team

* Devkaran Jawal (22BCE3048)
* Shambhavi Shree (22BCE2531)
* Dhevatha SP (22BCE0826)

## 🎯 The Problem

India faces significant road safety challenges from hazards like landslides, sharp turns, potholes, and sudden weather changes. This project addresses three key gaps in existing solutions:

1.  **Lack of Indian Road Datasets:** Most AI models are trained on Western data and perform poorly on unique Indian road conditions (unmarked roads, animals, etc.).
2.  **Poor Weather Performance:** Vision-only systems (cameras, LiDAR) fail in harsh conditions like heavy fog or rain.
3.  **High Cost:** Advanced sensors like LiDAR are too expensive for widespread deployment on rural or hilly roads.

## 💡 Proposed Solution

We built a hybrid **AI-Sensor Fusion System** that is cost-effective, robust, and designed for edge deployment.

Our system processes two streams of data simultaneously:
1.  **Vision Data (from Camera):** Analyzed by Deep Learning models (CNNs) to identify visual hazards.
2.  **Sensor Data (Synthetic):** Analyzed by Machine Learning models to detect anomalies in speed, vibration, and temperature.

A final **Fusion Model** (a CNN) combines the predictions from both streams to make a highly accurate and reliable final decision, reducing the false positives of a single-system approach.


## 🛠️ Technical Pipeline & Models

We implemented and compared multiple models to find the optimal pipeline.

1.  **Vision Models (Image Analysis):**
    * `MobileNet V2`: Lightweight, fast, but less accurate.
    * `ResNet50`: Deeper, more accurate feature extraction.

2.  **Sensor Models (Tabular Data Analysis):**
    * `Random Forest (RF)`
    * `Support Vector Machine (SVM)`
    * `XGBoost`

3.  **Final Hybrid Model:**
    The best-performing pipeline for this project was determined to be:
    **`MobileNet V2` (Vision) + `Random Forest` (Sensor) + `CNN` (Fusion)**

## 📊 Data Collection: A Pragmatic Approach

To overcome the "Lack of Indian Datasets" gap without a fleet of sensor-equipped vehicles, we used a **"Real-Image + Synthetic-Sensor"** strategy:

1.  **Real Images:** Thousands of real-world images for 7 hazard classes (pothole, landslide, animal crossing, etc.) were scraped from the web (`download_image.py`).
2.  **Synthetic Sensor Data:** For each image, we logically generated corresponding sensor data (`data_generator.py`). For example:
    * A "pothole" image was paired with a *high vibration* spike.
    * A "slippery road" image was paired with a *low temperature* reading.

This gave us a complete, paired multimodal dataset for training and testing our fusion pipeline.

## 🖥️ Demo

A Flask web application was built to serve as a proof-of-concept. The UI allows a user to upload an image and manually enter sensor readings (speed, vibration, temp) to get a real-time hazard analysis from the trained models.

