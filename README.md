# AI_internship

This repository contains documented concepts, code examples, and practical implementations related to PyTorch, Computer Vision, Image Segmentation, and segmentation frameworks.

---

## How to Run

To get started with this repository locally, clone the repository, install the required dependencies, and run the models. For example:

```bash
# Clone the repository
git clone https://github.com/AbiramiR-27/AI_internship.git
cd AI_internship

# Install the dependencies
pip install -r requirements.txt

# Run a sample inference (e.g., BiRefNet)
python 05-birefnet/inference.py --input sample.jpg
```

If you prefer to run the interactive notebooks in Google Colab, you can also use the **[Open in Colab]** badges provided at the top of each section below.

---

## Repository Structure

### 01 - PyTorch Basics
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbiramiR-27/AI_internship/blob/main/01-pytorch/pytorch_concepts.ipynb)

Topics Covered:
* Tensor Basics
* Tensor Operations
* Autograd
* Dataset and DataLoader
* Neural Networks
* Loss Functions
* Optimizers
* Training Loop
* Model Inference
* Introduction to Computer Vision Datasets (MNIST)

Key Learning Outcome:
Developed an understanding of the complete deep learning workflow using PyTorch.

---

### 02 - Object Segmentation and Open Source Libraries
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbiramiR-27/AI_internship/blob/main/02-object-segmentation/segmentation.ipynb)

Topics Covered:
* Computer Vision Fundamentals
* Image Segmentation Concepts
* Segmentation Masks
* Image Loading using OpenCV
* Grayscale Conversion
* Thresholding and Mask Generation
* Image Augmentation using Albumentations

Libraries Explored:
* OpenCV
* Albumentations
* Matplotlib

Key Learning Outcome:
Learned the fundamentals of image segmentation workflows, mask generation, image preprocessing, and augmentation techniques.

---

### 03 - SAM
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbiramiR-27/AI_internship/blob/main/03-sam/sam.ipynb)

Topics Covered:
- Segment Anything Model (SAM)
- Prompt-based Segmentation
- Mask Generation
- Segmentation Visualization

Key Learning Outcome:
Understood the workflow of prompt-based image segmentation and explored mask generation using the Segment Anything Model.

---

### 04 - MODNet
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbiramiR-27/AI_internship/blob/main/04-modnet/modnet.ipynb)

Topics Covered:
- Portrait Matting
- Human Segmentation
- Alpha Matte Generation
- Background Removal Workflow

Key Learning Outcome:
Understood portrait matting concepts and how human foregrounds can be extracted for background removal applications.

---

### 05 - BiRefNet
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbiramiR-27/AI_internship/blob/main/05-birefnet/birefnet.ipynb)

Topics Covered:
- Foreground Extraction
- Background Removal
- Boundary-Aware Segmentation
- Segmentation Workflow

Key Learning Outcome:
Understood foreground-background segmentation and the role of boundary-aware models in generating high-quality masks.

---

### 06 - Hugging Face Vision Fine-Tuning
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbiramiR-27/AI_internship/blob/main/06-huggingface-finetuning/hf.ipynb)

Topics Covered:
- Vision Transformers (ViT)
- Transfer Learning
- Fine-Tuning
- Trainer API
- Image Classification

**Fine-Tuning Takeaway:**
Fine-tuned a pretrained Vision Transformer (`google/vit-base-patch16-224`) checkpoint on the **CIFAR-10** image classification dataset (using a 500 training samples subset for computation efficiency). Over 1 epoch of training, the cross-entropy training loss decreased from **2.232** to **0.963** (a **~57% reduction**), achieving an evaluation loss of **1.106** at validation.

Key Learning Outcome:
Implemented a vision model fine-tuning workflow using Hugging Face Transformers and explored training, evaluation, and prediction pipelines.

---

### 07 - ISNet Segmentation (Refactored Pipeline)

Topics Covered:
- Configurable PyTorch Local Datasets and DataLoaders
- CNN Encoder-Decoder Segmentation Architectures
- Argparse configurations for hyperparameter tuning
- Side-by-side metric visualizers logged automatically to Weights & Biases (wandb)

Key Learning Outcome:
Transitioned prototype notebooks into robust, command-line executable Python files with local directory path settings and experiment tracking logs.

---

## Progress Status

- [x] PyTorch Basics
- [x] Object Segmentation Fundamentals
- [x] SAM / SAM2 Concepts
- [x] Hugging Face Vision Fine-Tuning
- [x] MODNet Concepts and Workflow
- [x] BiRefNet Concepts and Workflow
- [x] Refactored ISNet Training & Inference Pipeline

---

## Technologies Used

* Python
* PyTorch
* OpenCV
* Albumentations
* Matplotlib
* Weights & Biases (wandb)
* Google Colab
* GitHub
