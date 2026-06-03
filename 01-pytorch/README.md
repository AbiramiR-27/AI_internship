# PyTorch Basics and Computer Vision Introduction

## Overview

PyTorch is an open-source deep learning framework developed by Meta and widely used for building, training, and deploying machine learning and deep learning models.

---

# Objectives

The objective of this notebook is to understand:

* Tensor creation
* Tensor operations
* Autograd
* Dataset and DataLoader
* Neural Networks
* Loss Functions
* Optimizers
* Training Loop
* Model Inference
* Computer Vision datasets

---

# PyTorch Fundamentals

## Tensor Basics

A tensor is the primary data structure in PyTorch.

Examples:

* Scalar → Single value
* Vector → One-dimensional array
* Matrix → Two-dimensional array
* Tensor → Multi-dimensional array

Tensors are used to represent and process data in deep learning models.

---

## Tensor Operations

PyTorch supports various mathematical operations on tensors such as:

* Addition
* Subtraction
* Multiplication
* Division

These operations are performed efficiently and form the basis of neural network computations.

---

## Autograd

Autograd is PyTorch's automatic differentiation engine.

It automatically computes gradients required during backpropagation, allowing neural networks to learn and update their parameters during training.

---

## Dataset and DataLoader

### Dataset

A Dataset stores the training and testing data used by the model.

### DataLoader

A DataLoader loads data in batches and provides functionalities such as:

* Batch processing
* Data shuffling
* Efficient data loading

These utilities are essential when working with large datasets.

---

## Neural Networks

A neural network consists of interconnected layers that learn patterns from data.

In this notebook, a simple feed-forward neural network is implemented using:

* Linear Layers
* ReLU Activation Function

The network learns the relationship between input and output values through training.

---

## Loss Function

A loss function measures how far the model's predictions are from the actual target values.

The notebook uses:

### Mean Squared Error (MSE Loss)

MSE is commonly used for regression problems and helps evaluate model performance during training.

---

## Optimizer

Optimizers update model parameters based on computed gradients.

The notebook uses:

### Adam Optimizer

Advantages:

* Fast convergence
* Adaptive learning rates
* Widely used in deep learning applications

---

## Training Loop

The training loop consists of:

1. Forward Pass
2. Loss Calculation
3. Backpropagation
4. Parameter Update

This process is repeated for multiple epochs until the model learns the desired pattern.

---

## Model Inference

Inference refers to using a trained model to make predictions on unseen data.

After training, the model is tested with new input values to observe its predictive capability.

---

# Introduction to Computer Vision

Computer Vision is a field of Artificial Intelligence that enables computers to understand and process images and videos.

Applications include:

* Image Classification
* Object Detection
* Image Segmentation
* Face Recognition
* Medical Image Analysis

---

## torchvision

torchvision is a PyTorch library designed for computer vision tasks.

It provides:

* Popular image datasets
* Image transformations
* Pretrained models
* Utilities for image processing

---

## MNIST Dataset

The MNIST dataset contains grayscale images of handwritten digits from 0 to 9.

Dataset Information:

* 60,000 training images
* 10,000 testing images
* Image size: 28 × 28 pixels

MNIST is commonly used as an introductory dataset for learning computer vision workflows.

---

## Why Use MNIST?

MNIST helps in understanding:

* Image datasets
* Image tensor representations
* Dataset loading
* Data preprocessing
* Deep learning workflows for image data

---

# Learning Outcome

After completing this notebook, I gained practical understanding of:

* PyTorch fundamentals
* Tensor operations
* Automatic differentiation
* Neural network construction
* Model training workflow
* Dataset and DataLoader usage
* Computer vision dataset handling using torchvision

These concepts provide the foundation for building and training deep learning models for computer vision applications.

---

# Technologies Used

* Python
* PyTorch
* torchvision
* Google Colab
