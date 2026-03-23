# Drone vs Bird Detection & Segmentation

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-Web%20App-lightgrey.svg)
![YOLOv8/11](https://img.shields.io/badge/YOLO-v8%20%7C%20v11-yellow.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)

> **Course:** Image Processing and Applications (CS406.Q12)  
> **Instructor:** MSc. Cap Pham Dinh Thang

## Project Overview

This project focuses on the accurate object detection and instance segmentation of **Drones** and **Birds** using state-of-the-art Deep Learning models. Distinguishing between drones and birds is a critical challenge in airspace security, aviation safety, and wildlife monitoring. 

We have developed an end-to-end pipeline including data preprocessing, model training (YOLOv8 and YOLOv11 architectures), and building a user-friendly web application using Flask for real-time inference on images and videos.

### Key Features
- **Object Detection & Segmentation:** High-precision localization and pixel-level segmentation.
- **Multi-model Support:** Direct comparison setup between 4 variants (YOLOv8 Baseline, YOLOv8-Seg, YOLOv11 Baseline, YOLOv11-Seg).
- **Interactive Web Interface:** Drag-and-drop web application built with Flask for easy image and video testing.
- **Data-Centric Approach:** Proactively identified and automatically fixed mislabeled annotations, standardized classes, and removed duplicate data from the original academic dataset to enhance model reliability.

## Dataset

The original dataset is sourced from the scientific publication *"YOLO-based segmented dataset for drone vs bird detection" (Shandilya et al., 2023)*. 

We thoroughly cleaned this dataset, including correcting mislabels, standardizing to the format `0: Drone`, `1: Bird`, and removing all duplicate images.
- **Cleaned Dataset (Kaggle):** [Bird and Drone Dataset](https://www.kaggle.com/datasets/truong9/bird-and-drone)
- **Original Citation Source:** [Data in Brief Paper](https://www.sciencedirect.com/science/article/pii/S2352340923004742)

## Project Directory Structure

```text
.
├── docs/                           # Documentation and presentation slides
└── source/                         # Main source code directory
    ├── app.py                      # Backend Application (Flask)
    ├── requirements.txt            # Python dependencies list
    ├── templates/                  # HTML templates (index.html)
    ├── uploads/                    # Temporary directory for test images/videos
    ├── *_dataset.ipynb             # Notebooks for EDA, verification, and data cleaning
    ├── yolov8-baseline/            # Training & evaluation info for YOLOv8 Object Detection
    ├── yolov8-seg/                 # Training & evaluation info for YOLOv8 Instance Segmentation
    ├── yolov11-baseline/           # Training & evaluation info for YOLOv11 Object Detection
    └── yolov11-seg/                # Training & evaluation info for YOLOv11 Instance Segmentation
```

*(Note: The model weights files in `.pt` format are stored separately on Google Drive).*
- **Download Pre-trained Weights:** [Google Drive Link](https://drive.google.com/drive/folders/1sVy7wwddEWMQ-68ZCEKrVSCO9pXXZwBe?usp=drive_link)

## Installation and Usage

### System Requirements
- Python 3.8+
- Using a GPU (NVIDIA) is highly recommended to ensure real-time video stream inference without frame drops. If not available, the model will run on the CPU by default.

### Environment Setup

1. **Clone the repository and install dependencies:**
   ```bash
   cd source
   pip install -r requirements.txt
   ```

2. **Setup Model Weights Directory:**
   Please download all the weights from the Google Drive link above and map them correctly into the corresponding `weights/` directory inside, based on the `MODEL_PATHS` routing variable from the `app.py` file.

3. **Run the Local Web Application:**
   ```bash
   python app.py
   ```

4. **Test the App Interface:**
   Use a web browser to access: `http://127.0.0.1:5000`

## Retraining the Model (Training Guide)

To retrain the models according to your specific needs:
1. **Clean the Database:** Run the `Fix_dataset.ipynb` file to apply the patch fixing annotation and class errors.
2. **Start Training:** Access the desired model's folder (e.g., `yolov11-seg/`) and run its corresponding Jupyter notebook on environments like Google Colab / Kaggle. Remember to update the data paths in the YAML configuration.

## Authors / Team Members

- **Vo Dinh Trung** - Student ID: 22521571
- **Truong Phuc Truong** - Student ID: 22521587

---

