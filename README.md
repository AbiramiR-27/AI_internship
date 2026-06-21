# 🚀 AI Internship Repository: PyTorch, Computer Vision & Image Segmentation

Welcome to my **AI Internship Repository**! This repository is a comprehensive compilation of documented concepts, interactive notebooks, command-line pipelines, and production API deployments. It covers everything from PyTorch core mechanics to state-of-the-art vision models and background removal APIs.

---

## 🗺️ Project Navigation

Below is a summary of all milestones and projects in this repository:

| Module / Directory | Status | Resources & Interactive Badges | Key Focus Area |
| :--- | :---: | :---: | :--- |
| **[01-pytorch]** |  Completed | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbiramiR-27/AI_internship/blob/main/01-pytorch/pytorch_concepts.ipynb) | Tensors, Autograd, Custom Datasets, Loss Functions, Training Loops |
| **[02-object-segmentation]** |  Completed | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbiramiR-27/AI_internship/blob/main/02-object-segmentation/segmentation.ipynb) | Grayscale, Thresholding, Masking, Albumentations Augmentations |
| **[03-sam]** |  Completed | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbiramiR-27/AI_internship/blob/main/03-sam/sam.ipynb) | Zero-Shot Segmentation, Prompt-based Mask Generation (SAM/SAM 2) |
| **[04-modnet]** |  Completed | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbiramiR-27/AI_internship/blob/main/04-modnet/modnet.ipynb) | Portrait Matting, Alpha Matte, Background Removal |
| **[05-birefnet]** |  Completed | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbiramiR-27/AI_internship/blob/main/05-birefnet/birefnet.ipynb) | Salient Object Detection, Boundary-Aware Segmentation |
| **[06-huggingface-finetuning]** |  Completed | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbiramiR-27/AI_internship/blob/main/06-huggingface-finetuning/hf.ipynb) | Fine-tuning Vision Transformers (ViT) on CIFAR-10 Dataset |
| **[07-isnet]** |  Completed | *Command Line Pipeline* | Modular Training & Inference (IS-Net), W&B Tracking |
| **[Project]** |  Completed | *FastAPI Deployment* | Deployment of IS-Net Segmenter via FastAPI endpoints |

---

## ⚙️ How to Run

### 1. Local Setup
Clone this repository and install all global requirements:
```bash
# Clone the repository
git clone https://github.com/AbiramiR-27/AI_internship.git
cd AI_internship

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the FastAPI Serve Server (Production API)
The final production API resides inside the **[Project]** directory. It serves predictions using the custom trained IS-Net model.

Navigate to the `Project` folder, install requirements, and run the FastAPI server:
```bash
cd Project
pip install -r requirements.txt
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```
Once started, you can access:
* **Interactive API Documentation (Swagger UI)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **API Health Check**: `GET http://127.0.0.1:8000/`

#### Call the `/predict` Endpoint via curl:
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@test.jpg' --output result_mask.png
```

---

## 📂 Detailed Directory Overview

### 🏷️ 01 - PyTorch Basics
* **Interactive Notebook**: [pytorch_concepts.ipynb]
* **Highlights**: Tensor arithmetic, autograd computation, custom PyTorch Datasets/DataLoaders, loss functions, optimization loops, and training models from scratch on the MNIST digits dataset.
* **Outcome**: Solidified understanding of standard deep learning execution graphs in PyTorch.

### 🏷️ 02 - Object Segmentation and Open Source Libraries
* **Interactive Notebook**: [segmentation.ipynb]
* **Highlights**: Explores computer vision foundations. Implements mask thresholding via OpenCV (`cv2.threshold`), structural masking, and image augmentation workflows using `Albumentations` (pixel level shifts, translations, and flips).
* **Outcome**: Gained hands-on experience preprocessing image boundaries and engineering segmentations.

### 🏷️ 03 - SAM (Segment Anything Model)
* **Interactive Notebook**: [sam.ipynb]
* **Highlights**: Integrates Meta's SAM foundation model using the Hugging Face Transformers pipeline. Demos zero-shot point prompts, bounding box cues, and multi-mask creation.
* **Outcome**: Explored foundation vision models for generalizable foreground masking.

### 🏷️ 04 - MODNet (Portrait Matting)
* **Interactive Notebook**: [modnet.ipynb]
* **Highlights**: Portrait transparency matting. Explores boundary refinement, alpha matte estimation, and structural division branches (Semantic, Detail, Fusion) to isolate clean subject boundaries (e.g., hair details).
* **Outcome**: Built matting pipelines to strip backgrounds and save transparent foreground files.

### 🏷️ 05 - BiRefNet (Salient Object Detection)
* **Interactive Notebook**: [birefnet.ipynb]
* **Highlights**: High-resolution salient object detection. Uses boundary-aware loss formulations to detect and isolate prominent objects.
* **Outcome**: Developed code to perform automatic, zero-shot salient object extraction.

### 🏷️ 06 - Hugging Face Vision Fine-Tuning
* **Interactive Notebook**: [hf.ipynb]
* **Highlights**: Fine-tunes a Pretrained Vision Transformer (`google/vit-base-patch16-224`) on the CIFAR-10 image dataset using the Hugging Face Trainer API.
* **Performance takeaway**: 
  > [!NOTE]
  > Over 1 epoch of training, the cross-entropy training loss decreased from **2.232** to **0.963** (a **~57% reduction**), achieving an evaluation loss of **1.106** at validation.

### 🏷️ 07 - ISNet Segmentation (Refactored Pipeline)
* **Directory**: [07-isnet]
* **Highlights**: Refactored prototype notebooks into a structured, configurable local training/inference framework. Implements the IS-Net architecture utilizing nested Residual U-blocks (RSUs) and Multi-Loss Fusion for Dichotomous Image Segmentation (DIS).
* **Tracking**: Integrates Weights & Biases (`wandb`) to monitor training losses and log predicted segmentation masks side-by-side.

### 🏷️ Project (Production FastAPI Service)
* **Directory**: [Project]
* **Highlights**: Contains the finalized, production-ready system. Integrates the trained IS-Net model weights (`isnet_model.pth`) and hosts them behind a robust FastAPI service.
* **Modules**:
  * [api.py]: FastAPI server, `/predict` endpoint returning mask PNGs.
  * [train.py]: Script to re-train the model.
  * [inference.py]: Command line script for processing single images.

---

## 🛠️ Technologies & Libraries Used

* **Programming Language**: Python 3.10+
* **Deep Learning Framework**: PyTorch, Torchvision
* **Transformers & Vision Hub**: Hugging Face Transformers, Hugging Face Hub
* **API Framework**: FastAPI, Uvicorn (ASGI Server)
* **Computer Vision & Image Processing**: OpenCV, Pillow (PIL), Albumentations, scikit-image, scipy
* **Visualization & Logging**: Matplotlib, Weights & Biases (wandb)

---

