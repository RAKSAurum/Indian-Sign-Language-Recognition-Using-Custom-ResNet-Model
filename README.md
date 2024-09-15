# Indian Sign Language Recognition Using Custom ResNet Model

This project involves the development of a custom ResNet model to recognize Indian Sign Language from image data. The dataset used contains images of various signs, and the model is designed to classify these signs into their respective categories. This README provides an overview of the project, including setup, usage, and results.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Setup](#setup)
4. [Model Architecture](#model-architecture)
5. [Training and Evaluation](#training-and-evaluation)
6. [Results](#results)
7. [Visualizations](#visualizations)
8. [Future Work](#future-work)
9. [License](#license)

## Project Overview

This project aims to build a robust model for recognizing Indian Sign Language gestures using deep learning. The model is built on a custom ResNet architecture with various enhancements to improve accuracy and generalization.

## Dataset

The dataset used is the [Indian Sign Language Dataset](https://www.kaggle.com/datasets/atharvadumbre/indian-sign-language-islrtc-referred). It consists of images of hand signs representing different characters. The data is organized into training and testing sets.

## Setup

### Prerequisites

- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- scikit-image
- Pandas

### Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install required packages:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset:**

   Place the dataset JSON file (`kaggle.json`) in the root directory and run:

   ```bash
   kaggle datasets download -d atharvadumbre/indian-sign-language-islrtc-referred
   ```

4. **Unzip the dataset:**

   ```bash
   mkdir images
   unzip indian-sign-language-islrtc-referred.zip -d images
   ```

## Model Architecture

The model is based on a custom ResNet architecture built on top of the ResNet50 base model. Key features include:

- **Residual Blocks:** Custom residual blocks with dropout layers to improve model generalization.
- **Global Max Pooling:** Used instead of Global Average Pooling to reduce dimensionality.
- **Dense Layers:** Fully connected layers with dropout for regularization.

## Training and Evaluation

1. **Data Augmentation:**
   - Data is augmented using techniques like rotation, shifting, shearing, and flipping to enhance model robustness.

2. **Model Compilation:**
   - Optimizer: SGD
   - Loss Function: Categorical Crossentropy
   - Metrics: Accuracy, Precision, Recall, AUC, F1 Score

3. **Training:**
   - The model is trained for 20 epochs with early stopping and learning rate reduction callbacks.

4. **Evaluation:**
   - The model is evaluated on validation and test datasets to measure performance.

## Results

The trained model is saved in both `.h5` and `.keras` formats. Performance metrics on the validation and test datasets are provided. The results include loss, accuracy, AUC, precision, and F1 score.

## Visualizations

Plots for accuracy, loss, AUC, precision, and F1 score are generated to visualize model performance over epochs.

## Future Work

- Explore additional data augmentation techniques.
- Experiment with other pre-trained models or architectures.
- Perform hyperparameter tuning to further improve model performance.
