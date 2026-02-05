# Teeth Classification using CNN (Week 1)
## Project Overview
This project presents an end-to-end computer vision solution for **dental teeth image classification**.
The goal is to classify dental images into **7 distinct classes** using a **Convolutional Neural Network (CNN) built from scratch**.
This work represents **Week 1** of the project and focuses on:
- Data preprocessing
- Data visualization
- CNN baseline model training
The project is part of an AI-driven healthcare initiative aimed at improving diagnostic accuracy in dental applications.
---
## 🎯 Objectives
- Prepare dental images for training through **normalization and data augmentation**
- Visualize the **class distribution** to analyze dataset balance
- Display **images before and after augmentation**
- Build and train a **CNN model from scratch** using TensorFlow
- Establish a **baseline performance** for future improvements
---
## Project Structure
teeth-classification-cnn/
│
├── dataset/
│ └── (processed and augmented dental images)
│
├── notebook/
│ └── teeth_classification_week1.ipynb
│
├── model/
│ ├── teeth_classifier_model.h5
│ └── model_architecture.png
│
├── pdf_task/
│ └── Teeth Classification.pdf
│
└── README.md
> All preprocessing and training were performed using the Kaggle environment.
---
## Dataset & Preprocessing
- Dental images were resized and normalized to improve training stability
- Pixel values were scaled to the range **[0, 1]**
- Data augmentation techniques were applied to enhance generalization, including:
  - Rotation
  - Horizontal flipping
  - Zooming
These steps help reduce overfitting and improve the robustness of the model.
---
## Data Visualization
The notebook includes multiple visualizations to better understand the dataset and model behavior:
- **Class distribution plots** to analyze dataset balance
- **Original vs augmented image comparisons**
- **Training vs validation accuracy curves**
- **Training vs validation loss curves**
These visualizations confirm stable learning behavior and good generalization.
---
## Model Architecture
A **CNN model built from scratch** was implemented with the following components:
- Multiple `Conv2D` layers for feature extraction
- `MaxPooling2D` layers for spatial downsampling
- `GlobalAveragePooling2D` to reduce parameters and prevent overfitting
- Fully connected (`Dense`) layers for classification
- `Dropout` layers for regularization
The following diagram illustrates the CNN architecture used for teeth classification:
![Model Architecture](images/model_architecture.png)
---
## Training Results
- The model achieved strong baseline performance:
  - High training and validation accuracy
  - Smooth convergence of loss curves
  - No significant overfitting observed
- Minor validation fluctuations are expected due to data augmentation
This baseline model will be used as a reference point for future improvements.
The following plot shows the training and validation accuracy over epochs:
![Training vs Validation Accuracy](images/training_validation_accuracy.png)
---
## Saved Model
The trained model is saved in the following path:
model/teeth_classifier_model.h5
### Load the model:
```python:disable-run
from tensorflow.keras.models import load_model
model = load_model("model/teeth_classifier_model.h5")
```
