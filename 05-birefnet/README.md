# BiRefNet

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbiramiR-27/AI_internship/blob/main/05-birefnet/birefnet.ipynb)

## Overview

BiRefNet (Bilateral Reference Network) is a segmentation model designed for foreground extraction and high-quality background removal.

The model focuses on generating accurate segmentation masks while preserving fine object boundaries.

---

## Key Features

* Foreground extraction
* Background removal
* Fine boundary prediction
* High-resolution segmentation
* Accurate object masks

---

## Workflow

Input Image
↓
BiRefNet
↓
Segmentation Mask
↓
Foreground Extraction
↓
Background Removal

---

## Applications

* Product photography
* E-commerce
* Image editing
* Content creation
* Object extraction

---

## Advantages

* Better boundary preservation
* High-quality masks
* Suitable for complex objects
* Effective foreground-background separation

---

## Learning Outcome

* Understood foreground segmentation concepts
* Explored mask generation workflows
* Learned the role of boundary-aware segmentation models

---

## How to Run

To run a sample inference using BiRefNet:

```bash
# Clone the repository (if you haven't already)
git clone https://github.com/AbiramiR-27/AI_internship.git
cd AI_internship

# Install the required dependencies
pip install -r requirements.txt

# Run the inference script with the sample input
python 05-birefnet/inference.py --input sample.jpg
```

