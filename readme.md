# FeDR: Federated Learning for Road Segmentation

This repository contains the code for FeDR, a federated learning framework designed for road segmentation in autonomous driving using Vision Transformers (ViT) and Convolutional Neural Networks (CNNs). 

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Inference](#inference)
- [Results](#results)

## Introduction
FeDR is a privacy-preserving federated learning framework that enables multiple organizations to collaboratively train a road segmentation model without sharing raw data. The model combines Vision Transformers (ViTs) for global context capture and Convolutional Neural Networks (CNNs) for local feature extraction.

## Installation

To get started with FeDR, follow these instructions:

### Prerequisites
- Python 3.8 or later
- PyTorch 1.10+
- CUDA (Optional, but recommended for GPU support)

### Clone the repository
```bash
git clone https://github.com/abdkhanstd/FedR
cd FedR
```

### Install the required dependencies
```bash
pip install -r requirements.txt
```

## Dataset Preparation
1. Download the KITTI road dataset and prepare the training and testing images.
2. Organize the dataset structure as follows:
    ```
    datasets/
    └── KITTIRoad/
        ├── training/
        │   ├── images/
        │   └── labels/
        └── testing/
            └── images/
    ```
3. Update the dataset paths in the configuration file `scripts/config.py`.

## Training
To train the federated model across multiple companies, run the following script:

```bash
python train.py
```

This will:
- Load the dataset for each company.
- Train the model locally for each company.
- Aggregate the models on the server side after each round.

### Parameters
You can modify the training parameters like batch size, number of epochs, and learning rate in `params.py`.

### Model Weights
Pretrained model weights will be saved under `weights/`. You can resume training from saved models if needed.

## Inference
To evaluate the model on the test dataset:

```bash
python inference.py
```

This script will:
- Load the trained model.
- Run inference on the test images.
- Output the road segmentation results for each image.

## Results
For quantitative results of the trained model on the KITTI dataset, refer to:
- [KITTI Benchmark Results](https://www.cvlibs.net/datasets/kitti/eval_road_detail.php?result=7be66e7836f2bd6559126d1a025a5395da80eab4)

You can also find the code for the proposed framework at:
- [GitHub Repository](https://github.com/abdkhanstd/FedR)

--- 

