# Project Reference Guide: File-by-File Documentation

This document provides a detailed, file-by-file explanation of your entire internship repository. It outlines the purpose, code contents, and roles of each file inside all milestone folders (`01` through `07`).

---

## Workspace Directory Tree and File Catalog

Below is the complete catalog of files in your repository, categorized by directory.

### Root Directory
*   **[README.md](file:///c:/Users/HP/OneDrive/Desktop/intern/AI_internship/README.md)**: Main repository README file. It describes the overall internship project goals, milestone tasks, and directories.
*   **[requirements.txt](file:///c:/Users/HP/OneDrive/Desktop/intern/AI_internship/requirements.txt)**: Global dependencies file for the parent repository, listing basic deep learning libraries (`torch`, `torchvision`, `numpy`, `matplotlib`, `wandb`).
*   **[PROJECT_DOCUMENTATION.md](file:///c:/Users/HP/OneDrive/Desktop/intern/AI_internship/PROJECT_DOCUMENTATION.md)**: This document. Serves as the central reference guide for understanding all codes, concepts, and files.

---

### Folder: `01-pytorch` (PyTorch Basics)
*   **[pytorch_concepts.ipynb](file:///c:/Users/HP/OneDrive/Desktop/intern/AI_internship/01-pytorch/pytorch_concepts.ipynb)**:
    *   *Purpose*: Educational Jupyter notebook to demonstrate fundamental PyTorch concepts.
    *   *Code Contents*: Contains Python code cells for tensor allocation, tensor math operators, autograd gradients, local DataLoader configurations, neural network sequential blocks, mean-squared error loss calculation, Adam optimizer steps, a 50-epoch training loop, scalar predictions, and loading/transforming MNIST digits.
*   **[README.md](file:///c:/Users/HP/OneDrive/Desktop/intern/AI_internship/01-pytorch/README.md)**:
    *   *Purpose*: Module documentation. Explains PyTorch installation, basic tensor definitions, and objectives.

---

### Folder: `02-object-segmentation` (Segmentation Fundamentals)
*   **[segmentation.ipynb](file:///c:/Users/HP/OneDrive/Desktop/intern/AI_internship/02-object-segmentation/segmentation.ipynb)**:
    *   *Purpose*: Demonstrates baseline computer vision segmentations and data augmentations.
    *   *Code Contents*: Contains code cells using OpenCV (`cv2`) to read images, translate them to grayscale, apply binary threshold segmentations (`cv2.threshold`), and execute data augmentations (flips, contrast shifts) using the `Albumentations` library, followed by matplotlib rendering functions.
*   **[README.md](file:///c:/Users/HP/OneDrive/Desktop/intern/AI_internship/02-object-segmentation/README.md)**:
    *   *Purpose*: Module documentation. Explains thresholding algorithms and the role of data augmentation.

---

### Folder: `03-sam` (Segment Anything Model)
*   **[sam.ipynb](file:///c:/Users/HP/OneDrive/Desktop/intern/AI_internship/03-sam/sam.ipynb)**:
    *   *Purpose*: Demonstrates mask generation using pre-trained foundation models.
    *   *Code Contents*: Contains code cells installing SAM 2, importing pipelines from huggingface transformers, generating image masks automatically, printing generated statistics, and overlaying segmentation masks on raw target images with transparency.
*   **[Screenshot 2026-06-04 213334.png](file:///c:/Users/HP/OneDrive/Desktop/intern/AI_internship/03-sam/Screenshot%202026-06-04%20213334.png)**:
    *   *Purpose*: Sample test image. Used as the input file for the SAM segmentation notebook.
*   **[README.md](file:///c:/Users/HP/OneDrive/Desktop/intern/AI_internship/03-sam/README.md)**:
    *   *Purpose*: Module documentation. Outlines pre-trained zero-shot segmentation workflows.

---

### Folder: `04-modnet` (Portrait Matting)
*   **[modnet.ipynb](file:///c:/Users/HP/OneDrive/Desktop/intern/AI_internship/04-modnet/modnet.ipynb)**:
    *   *Purpose*: Visualizes portrait matting pipeline stages.
    *   *Code Contents*: Contains PIL code cells to display portrait images, prints schematic diagrams of semantic/detail/fusion branches, and defines list dictionaries representing input image and alpha matte matching mockup.
*   **[README.md](file:///c:/Users/HP/OneDrive/Desktop/intern/AI_internship/04-modnet/README.md)**:
    *   *Purpose*: Module documentation. Details transparency matting theories and lists the trade-offs of the MODNet model.

---

### Folder: `05-birefnet` (Salient Object Detection)
*   **[birefnet.ipynb](file:///c:/Users/HP/OneDrive/Desktop/intern/AI_internship/05-birefnet/birefnet.ipynb)**:
    *   *Purpose*: Visualizes salient segmentations.
    *   *Code Contents*: Contains code cells plotting input images and corresponding ground truth masks side-by-side using PIL, prints structural stages, and sets up input list mockups.
*   **[README.md](file:///c:/Users/HP/OneDrive/Desktop/intern/AI_internship/05-birefnet/README.md)**:
    *   *Purpose*: Module documentation. Explains salient object extraction, background removal, and details local CLI "How to Run" execution snippets along with Google Colab notebook badges.

---

### Folder: `06-huggingface-finetuning` (ViT Classification)
*   **[hf.ipynb](file:///c:/Users/HP/OneDrive/Desktop/intern/AI_internship/06-huggingface-finetuning/hf.ipynb)**:
    *   *Purpose*: Fine-tunes Vision Transformers on a classification task.
    *   *Code Contents*: Contains code cells installing requirements, loading CIFAR-10, setting up custom subsets, importing pre-trained image processors/models, creating collate functions, setting `TrainingArguments` (epochs, batch sizes), runs `Trainer.train()`, evaluates accuracy, and plots outputs using matplotlib.
*   **[README.md](file:///c:/Users/HP/OneDrive/Desktop/intern/AI_internship/06-huggingface-finetuning/README.md)**:
    *   *Purpose*: Module documentation. Outlines Hugging Face Vision Transformer fine-tuning workflows, and documents quantifiable training/validation performance metrics (loss reduction from 2.232 to 0.963, representing ~57.2% reduction) on the CIFAR-10 dataset.


---

### Folder: `07-isnet` (Custom Segmentation Pipeline)
This is the central production folder containing the main refactored image segmentation pipeline.

*   **[dataset.py](file:///c:/Users/HP/OneDrive/Desktop/intern/AI_internship/07-isnet/dataset.py)**:
    *   *Purpose*: Handles data parsing and batch loader creation.
    *   *Code Contents*: Implements `LocalSegmentationDataset`, which inherits from `torch.utils.data.Dataset` to scan customizable paths (looking for `images` and `masks` directories) and matches filenames. It also defines `get_dataloader` with target transformations that resize inputs to $256 \times 256$ and map masks strictly to binary float tensors (`(x > 0.5).float()`).
*   **[model.py](file:///c:/Users/HP/OneDrive/Desktop/intern/AI_internship/07-isnet/model.py)**:
    *   *Purpose*: Defines the neural network structure.
    *   *Code Contents*: Implements `ISNetDIS` inheriting from `nn.Module`. Contains a deep 6-stage encoder-decoder pipeline using nested Residual U-blocks (RSU-7, RSU-6, RSU-5, RSU-4, RSU-4F) and side outputs for multi-resolution intermediate supervision.
*   **[train.py](file:///c:/Users/HP/OneDrive/Desktop/intern/AI_internship/07-isnet/train.py)**:
    *   *Purpose*: Controls the model training loop and tracking metrics.
    *   *Code Contents*: Configures argparse flags for CLI training. Includes dataset path validation, initializes Weights & Biases (`wandb.init`), iterates over epochs, runs forward/backward passes, saves the state dictionary as `isnet_model.pth`, and logs side-by-side image visualizations to the W&B dashboard.
*   **[inference.py](file:///c:/Users/HP/OneDrive/Desktop/intern/AI_internship/07-isnet/inference.py)**:
    *   *Purpose*: Runs prediction on single images.
    *   *Code Contents*: Parses CLI arguments, loads trained `isnet_model.pth` weights on the appropriate device, runs a forward pass under evaluation mode without calculating gradients (`torch.no_grad()`), and saves the grayscale predicted mask to local disk.
*   **[README.md](file:///c:/Users/HP/OneDrive/Desktop/intern/AI_internship/07-isnet/README.md)**:
    *   *Purpose*: Outlines setup, dataset organization, and train/inference execution commands.
*   **[requirements.txt](file:///c:/Users/HP/OneDrive/Desktop/intern/AI_internship/07-isnet/requirements.txt)**:
    *   *Purpose*: Local dependencies file (Torch, Torchvision, Numpy, Pillow, Matplotlib, Wandb).
*   **[observations.md](file:///c:/Users/HP/OneDrive/Desktop/intern/AI_internship/07-isnet/observations.md)**:
    *   *Purpose*: Records training loss values per epoch.
*   **[test.jpg](file:///c:/Users/HP/OneDrive/Desktop/intern/AI_internship/07-isnet/test.jpg)**:
    *   *Purpose*: Sample test image. Used for verify script inference.
