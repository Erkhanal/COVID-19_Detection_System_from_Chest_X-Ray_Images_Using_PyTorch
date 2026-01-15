# COVID-19 Detection System from Chest X-Ray Images Using PyTorch

## Table of Contents
- [Introduction](#introduction)
- [Objective](#objective)
- [Data Source](#data-source)
- [System Architecture](#system-architecture)
- [Steps for System Development](#steps-for-system-development)
- [Testing and Evaluation](#testing-and-evaluation)
- [Challenges Faced and Lessons Learned](#challenges-faced-and-lessons-learned)
- [Conclusion](#conclusion)
- [Technologies Used](#technologies-used)
- [References](#references)
- [Contact Information](#contact-information)

## Introduction 
<div align="justify"> This project develops an AI-based system to detect COVID-19 from chest X-ray images using PyTorch. Chest X-rays serve as a key diagnostic tool for identifying respiratory illnesses, including viral pneumonia and COVID-19. Manual diagnosis can be time-consuming and prone to errors due to the subtle differences in X-ray images. Traditional computer-aided approaches often face challenges with accuracy and generalization.</div><br>

<div align="justify"> To address these challenges, deep learning is leveraged with a pretrained ResNet-18 model to automatically classify chest X-rays into three categories: Normal, Viral Pneumonia, and COVID-19. This method enhances the efficiency, accuracy, and accessibility of COVID-19 detection. </div>

## Objective 
<div align="justify">

The main objective of this project is to build a **robust deep learning model** capable of accurately classifying chest X-ray images into three classes:

- Normal  
- Viral Pneumonia  
- COVID-19  

The goal is to develop an AI-based diagnostic tool that assists medical professionals in rapid and reliable COVID-19 detection.</div>

## Data Source  
<div align="justify"> 

**Data Source:** [COVID-19 Radiography Dataset - Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)  

- **Content:** Chest X-ray images of Normal, Viral Pneumonia, and COVID-19 patients  
- **File Format:** PNG images  
- **Structure:** Images were reorganized into training and test sets with class-wise directories. </div>

## System Architecture
<div align="justify"> 

The system uses a **Convolutional Neural Network (CNN)** architecture based on **ResNet-18**, a deep residual network pretrained on ImageNet. The architecture includes:

-  **Input Layer:** Resized X-ray images (224×224)  
-  **Convolutional Layers:** Feature extraction using residual blocks  
-  **Fully Connected Layer:** Modified to output 3 classes  
-  **Softmax Layer:** For probability prediction of each class  

**Workflow:**

- **Data Preprocessing:** Resizing, normalization, and augmentation  
- **Training:** Using Cross-Entropy loss and Adam optimizer  
- **Evaluation:** Accuracy and loss metrics on the test set  
- **Visualization:** Display of images with true and predicted labels 

## Steps for System Development
<div align="justify">

**Step 1: Dataset Preparation**  
- Organize the dataset into class-wise directories for training and testing  
- Randomly select 30 images per class for testing  

**Step 2: Data Loading**  
- Implement a **custom PyTorch Dataset** class (`ChestXRayDataset`)  
- Apply transformations: Resize, ToTensor, Normalization, Random Horizontal Flip (for training)  

**Step 3: DataLoader Setup**  
- Create PyTorch DataLoaders for batching and shuffling data  
- Batch size set to 6 for both training and testing  

**Step 4: Model Selection and Preparation**  
- Load **pretrained ResNet-18** from torchvision  
- Replace the final fully connected layer to output 3 classes  

**Step 5: Training**  
- Use **Cross-Entropy Loss** and **Adam optimizer**  
- Train for 1 epoch (can be extended for better performance)  
- Include **periodic evaluation** and visualization of predictions  

**Step 6: Evaluation and Visualization**  
- Calculate validation loss and accuracy  
- Visualize predictions with correct predictions in green and misclassifications in red  </div>

## Testing and Evaluation
<div align="justify">

- Evaluate the trained model on the test set using accuracy and loss  
- Sample predictions were visualized to qualitatively assess model performance  
- Early stopping implemented if accuracy exceeded 95%  

## Challenges Faced and Lessons Learned

**Limited Data per Class:**  
   - COVID-19 X-ray images are fewer than other classes, making data augmentation essential.  

**Model Adaptation:**  
   - Pretrained models required careful modification of the final layer for multi-class classification.  

**Training Stability:**  
   - Small batch size and low learning rate were necessary to prevent overfitting.  

**Visualization for Debugging:**  
   - Visualizing batches helped ensure the model was learning meaningful features. </div>

## Conclusion
<div align="justify"> 

This project demonstrates an AI-based system for **rapid COVID-19 detection from chest X-rays**. Using PyTorch and a pretrained ResNet-18, the system achieved reliable classification performance. This approach can **assist medical professionals** in faster diagnosis and improve clinical decision-making.</div>

## Technologies Used
- Python 3  
- PyTorch  
- Torchvision  
- NumPy  
- PIL (Python Imaging Library)  
- Matplotlib 

## References
- COVID-19 Radiography Dataset - Kaggle
- Python documentations
- PyTorch Documentation  
- Torchvision Documentation
- Stack Overflow

## Contact Information
Created by https://github.com/Erkhanal - feel free to contact !
