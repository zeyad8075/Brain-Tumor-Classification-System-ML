<h1>Brain Tumor Classification System using Deep Learning</h1>

This repository contains a complete end-to-end deep learning system for brain tumor image classification based on medical MRI images.
The project focuses on training a convolutional neural network (CNN) model to accurately classify brain tumor types and deploying the trained model within a user-friendly interface for real-time prediction.

<h1>Repository Contents</h1>

The repository includes the following main components:

🔹 1. Model Training Code

Preprocessing and normalization of brain MRI images

Training a deep learning classification model (CNN-based architecture)

Model evaluation using accuracy and loss metrics

Saving the trained model for inference

🔹 2. Trained Model

A pre-trained deep learning model ready for inference

Stored in a serialized format for easy loading and deployment

🔹 3. Final User Interface

A graphical interface that allows users to:

Upload a brain MRI image

Run the trained model on the uploaded image

Display the predicted tumor class along with relevant classification information

<h1>System Workflow</h1>

The user uploads a brain MRI image through the interface

The image is preprocessed automatically

The trained model performs tumor classification

The predicted tumor type and related information are displayed to the user

<h1>Project Objective</h1>

The goal of this project is to demonstrate how deep learning can be applied to medical image analysis, particularly in assisting the classification of brain tumors, while providing a practical and deployable solution suitable for academic and research purposes.

<h1>Dataset</h1>
Images: The dataset includes MRI scans, which are categorized based on the presence of different types of brain tumors, such as glioma, meningioma, and pituitary tumors. The images are available in JPG format

Labels: Each image is labeled with the corresponding tumor type.

Number of Samples: The dataset consists of 18279 images across 4 classes (tumor types).

https://www.kaggle.com/datasets/rishiksaisanthosh/brain-tumour-classification
