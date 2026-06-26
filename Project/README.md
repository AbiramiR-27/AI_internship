# 🛠️ Production ISNet Segmentation API

This directory contains the final production-ready system for the **IS-Net (Intermediate Supervision Network)** model. It includes a high-performance **FastAPI** web server for real-time background removal / mask generation, alongside CLI scripts for model training and local image inference.

To make the inference system independent of the training environment, the codebase is structured into two separate, self-contained directories: `training/` and `inference/`.

---

## 📌 Table of Contents
1. [Codebase Overview](#-codebase-overview)
2. [Environment Setup](#-environment-setup)
3. [Training Details](#-training-details)
4. [Running the FastAPI Server](#-running-the-fastapi-server)
5. [Docker Deployment](#-docker-deployment)
6. [Dataset Preparation & Custom Training](#-dataset-preparation--custom-training)
7. [Local Inference Script](#-local-inference-script)

---

## 📂 Codebase Overview

The project has been separated into independent training and inference directories:

### 🏋️ Training Directory (`Project/training/`)
*   **[train.py](training/train.py)**: CLI training script with Weights & Biases (`wandb`) experiment tracking.
*   **[dataset.py](training/dataset.py)**: PyTorch custom dataset loader (`DISDataset`) scanning local image/mask dirs.
*   **[model.py](training/model.py)**: Core **ISNetDIS** architecture definition for training.
*   **[requirements.txt](training/requirements.txt)**: Python package requirements for training (includes `wandb`, etc.).

### 🚀 Inference Directory (`Project/inference/`)
*   **[api.py](inference/api.py)**: FastAPI prediction server.
*   **[inference.py](inference/inference.py)**: CLI script to run segmentation inference on a local image.
*   **[model.py](inference/model.py)**: Core **ISNetDIS** architecture definition for inference.
*   **[isnet_model.pth](inference/isnet_model.pth)**: Saved model weights (approx. 176MB).
*   **[requirements-inference.txt](inference/requirements-inference.txt)**: Lightweight, inference-specific requirements file (CPU-only by default for fast Docker containerization).
*   **[Dockerfile](inference/Dockerfile)**: Docker instructions to build and deploy the inference FastAPI container.
*   **[results/](inference/results)**: Directory with test images (e.g. `test (1).jpg`).

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

Install the requirements depending on your task:

*   **For Inference & Deployment:**
    ```bash
    cd inference
    pip install -r requirements-inference.txt
    ```
*   **For Training & Fine-Tuning:**
    ```bash
    cd training
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

Start the ASGI web server using **Uvicorn** from the `inference/` folder:
```bash
cd Project/inference
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

## 🐳 Docker Deployment

To build and run the inference API container:

### 1. Build the Docker Image
Navigate to the `inference` directory and run:
```bash
cd Project/inference
docker build -t isnet-inference .
```

### 2. Run the Docker Container
Start the container and map port 8000:
```bash
docker run -p 8000:8000 --name isnet-container isnet-inference
```

### 3. Test Containerized API
You can test the endpoint using `curl` or browse to the container's interactive documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@results/test (1).jpg" \
     --output predicted_mask.png
```

---

## 📊 Dataset Preparation & Custom Training

To run custom training using [train.py], organize your local dataset folder inside `Project/training/` to match the format expected by the `DISDataset` class in [dataset.py]:

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
cd Project/training
python train.py --dataset_path /path/to/dataset_root --epochs 100 --batch_size 8
```
After training finishes, the updated weights will overwrite `isnet_model.pth`.

---

## 💻 Local Inference Script

To quickly perform offline inference on a single image without starting the FastAPI web server, use [inference.py] from the `inference/` folder:
1. Ensure your target image is placed in the directory (e.g. `results/test (1).jpg`).
2. Run the script:
   ```bash
   cd Project/inference
   python inference.py --image_path "results/test (1).jpg"
   ```
3. The segmented mask output will be saved as `predicted_mask.png`.

---
