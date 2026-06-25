# 🛠️ Production ISNet Segmentation API

This directory contains the final production-ready system for the **IS-Net (Intermediate Supervision Network)** model. It includes a high-performance **FastAPI** web server for real-time background removal / mask generation, alongside CLI scripts for model training and local image inference.

---

## 📌 Table of Contents
1. [Codebase Overview](#-codebase-overview)
2. [Environment Setup](#-environment-setup)
3. [Training Details](#-training-details)
4. [Running the FastAPI Server](#-running-the-fastapi-server)
5. [Dataset Preparation & Custom Training](#-dataset-preparation--custom-training)
6. [Local Inference Script](#-local-inference-script)
---

## 📂 Codebase Overview

The project consists of the following components:

*   **[api.py]**: The core API entrypoint. Wraps the model using FastAPI, preprocesses uploaded images, runs forward passes, and returns transparent/grayscale alpha masks.
*   **[model.py]**: The official **ISNetDIS** neural network architecture utilizing nested Residual U-blocks (RSUs) and Multi-Loss Fusion for Dichotomous Image Segmentation (DIS).
*   **[dataset.py]**: Contains the `DISDataset` PyTorch loader implementation that scans image and mask directories.
*   **[train.py]**: Standard CLI training script to fine-tune/re-train ISNet with Weights & Biases (`wandb`) experiment tracking.
*   **[inference.py]**: Simpler offline testing script to run inference on a local image.
*   **[requirements.txt]**: List of Python packages required for execution.

---

## ⚙️ Environment Setup

Ensure you are inside the `Project` directory:
```bash
cd Project
```

It is recommended to run the project in a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\activate

# Activate virtual environment (Linux/macOS)
source venv/bin/activate
```

Install the requirements:
```bash
pip install -r requirements.txt
```

---

## Training Details

Training Type:
- Trained from scratch

Dataset:
- DIS-5K 

Hardware:
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU

Training Hyperparameters:
- Epochs: 100
- Optimizer: Adam
- Learning Rate: 1e-4
- Batch Size: 2
- Image Size: 512 × 512

Training Duration:
- Approx 7 hours 20 minutes

Training Results:
- Initial Loss: ~2.218
- Final Loss: ~0.385

Observation:
- The loss decreased consistently throughout training, indicating stable convergence of the ISNet model on the DIS-5K dataset.

---
## 🌐 Running the FastAPI Server

Start the ASGI web server using **Uvicorn**:
```bash
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

### 🔍 Interactive API Documentation
Once the server is running, navigate to:
*   **Swagger UI Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) (Allows testing endpoints interactively)
*   **ReDoc Docs**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
*   **Health Check Endpoint**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

### 🚀 API Endpoints

#### 1. Home / Health Check
*   **Method**: `GET`
*   **Path**: `/`
*   **Description**: Validates that the model loaded correctly and the service is healthy.
*   **Response**:
    ```json
    {
      "message": "ISNet Segmentation API is running"
    }
    ```

#### 2. Image Segmentation Inference
*   **Method**: `POST`
*   **Path**: `/predict`
*   **Content-Type**: `multipart/form-data`
*   **Request Body**:
    *   `file` (Binary File): Upload the image you want to segment.
*   **Response**: A raw binary image stream (`image/png`) containing the extracted mask matching the original image dimensions.

**Input:**
- JPG / JPEG / PNG image

**Output:**
- PNG segmentation mask
---

## 📊 Dataset Preparation & Custom Training

To run custom training using [train.py], organize your local dataset folder to match the format expected by the `DISDataset` class in [dataset.py]:

```
dataset_root/
├── im/
│   ├── sample1.jpg
│   ├── sample2.png
│   └── ...
└── gt/
    ├── sample1.png
    ├── sample2.png
    └── ...
```

### Start Training Run:
Ensure you are logged in to Weights & Biases:
```bash
wandb login
```
Run the training CLI script:
```bash
python train.py --dataset_path /path/to/dataset_root --epochs 100 --batch_size 8
```
After training finishes, the updated weights will overwrite `isnet_model.pth`.

---

## 💻 Local Inference Script

To quickly perform offline inference on a single image without starting the FastAPI web server, use [inference.py]:
1. Place your target image in the directory (e.g. named `test(1).jpg`).
2. Run the script:
   ```bash
   python inference.py
   ```
3. The segmented mask output will be saved as `predicted_mask.png`.

---
