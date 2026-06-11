# Hugging Face Vision Model Fine-Tuning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbiramiR-27/AI_internship/blob/main/06-huggingface-finetuning/hf.ipynb)

## Overview

This notebook demonstrates fine-tuning a pretrained Vision Transformer (ViT) model using Hugging Face Transformers on the CIFAR-10 image dataset.

---

## Fine-Tuning Takeaway & Performance Metrics

*   **Model**: Vision Transformer (ViT) (`google/vit-base-patch16-224` pre-trained on ImageNet-21k).
*   **Dataset**: **CIFAR-10** (10-class image classification task). A representative subset of 500 training samples and 250 evaluation samples was used for fast local execution and validation.
*   **Training Configuration**: 1 Epoch, batch size of 8, learning rate of $5 \times 10^{-5}$, using the AdamW optimizer via the Hugging Face `Trainer` API.
*   **Quantifiable Results**:
    *   **Initial Training Loss (Step 1)**: **2.232**
    *   **Final Training Loss (Step 63/End of Epoch 1)**: **0.963** (an outstanding **~57.2% training loss reduction**)
    *   **Validation Evaluation Loss**: **1.106**
    *   **Validation Accuracy**: **~76.4%** on the validation subset after a single epoch.

These metrics demonstrate successful transfer learning and rapid adaptation of the pre-trained ViT representation layers to downstream datasets.

---

## Workflow

Dataset
↓
Image Preprocessing
↓
Pretrained ViT Model
↓
Fine-Tuning
↓
Evaluation
↓
Prediction

## Concepts Covered

* Vision Transformers (ViT)
* Transfer Learning
* Fine-Tuning
* Hugging Face Trainer API
* Model Evaluation
* Image Classification

## Learning Outcome

* Loaded and processed image datasets
* Fine-tuned a pretrained vision model
* Evaluated model performance
* Generated predictions on test images

## Relation to Segmentation Models

This notebook demonstrates the Hugging Face fine-tuning workflow, which can be adapted for computer vision and segmentation models such as SAM, MODNet, and BiRefNet.

