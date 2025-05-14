# Comparative Analysis of AlexCapsNet and ResCapsNet on Singular and Distributed Computing



## 📌 Abstract

This project presents a comparative analysis of two hybrid capsule network architectures—**AlexCapsNet** and **ResCapsNet**—for MNIST digit classification. Both models integrate **Capsule Networks (CapsNets)** with popular CNN backbones:  
- **AlexCapsNet** extends an AlexNet-like architecture with capsule layers.  
- **ResCapsNet** integrates residual blocks from ResNet before capsules.  

Both models use **dynamic routing** and were trained on **single GPU** and **multi-GPU** setups using **PyTorch DDP (Distributed Data Parallel)** and **mixed-precision training**.

---

## 🧠 Introduction

Capsule Networks (CapsNets) capture spatial hierarchies better than traditional CNNs but are computationally heavy.  
To mitigate this:
- **AlexCapsNet** uses deep convolutional layers from AlexNet before capsule layers.
- **ResCapsNet** leverages residual connections to maintain gradient flow in deeper architectures.

This project evaluates both in terms of **accuracy**, **convergence**, **model complexity**, and **scalability**.

---

## 🏗️ Architectures

### 🔹 AlexCapsNet
- **Input**: 28×28 grayscale images
- **Convolutional Backbone**: 5 convolutional layers with ReLU and max-pooling
- **Primary Capsules**: 32 capsules with 8D vectors
- **Digit Capsules**: 10 capsules, 16D each, dynamic routing
- **Loss**: Capsule margin loss

### 🔹 ResCapsNet
- **Input Resized**: 28×28 → 78×78
- **Initial Conv Layer**: Conv2d(1→32)
- **Residual Blocks**: 3 skip-connected blocks
- **Primary Capsules**: 1296 capsules, each 8D
- **Digit Capsules**: 10 capsules, 16D each
- **Loss**: Capsule margin loss

---

## ⚙️ Training Configuration

- **Optimizer**: Adam (lr=0.001)
- **Batch Size**: 32 (per GPU)
- **Epochs**: 20–50
- **Data Augmentation**: ±10° rotation
- **Learning Rate Scheduler**: StepLR (γ=0.7 every 2 epochs)
- **Routing Iterations**: 3
- **Precision**: FP16 (with GradScaler for stability)
- **Distributed**: PyTorch DDP for multi-GPU setup

---

## 🧪 Results

| Model           | Setup       | Test Accuracy | Epochs |
|----------------|-------------|---------------|--------|
| AlexCapsNet    | Multi-GPU   | 96.28%        | 20     |
| AlexCapsNet    | Single-GPU  | 98.06%        | 10     |
| ResCapsNet     | Multi-GPU   | 97.86%        | 10     |
| ResCapsNet     | Single-GPU  | **98.70%**    | 20     |

### 📈 Observations
- **AlexCapsNet**: Faster convergence; deeper CNN; larger parameter count.
- **ResCapsNet**: Higher final accuracy; compact; better for resource-limited settings.
- Both models performed well under DDP and single-GPU training.

---

## 📊 Visual Results

- **Confusion Matrices**: Near-perfect classification on MNIST
- **ROC Curves**: One-vs-Rest curves close to ideal for all classes
- Included for:  
  - AlexCapsNet (Single & Multi-GPU)  
  - ResCapsNet (Single & Multi-GPU)

---

## 📌 Comparative Analysis

| Metric              | AlexCapsNet                     | ResCapsNet                        |
|---------------------|----------------------------------|-----------------------------------|
| Convergence Speed   | Faster                          | Slightly Slower                   |
| Final Accuracy      | Slightly Lower                  | Slightly Higher                   |
| Complexity          | High (millions of params)       | Lower (few million params)        |
| Training Speed      | Slower per epoch                | Faster per epoch                  |
| Distributed Benefit | Significant                     | Moderate                          |
| Suitability         | High-resources, fast training   | Low-resources, high accuracy      |

---


