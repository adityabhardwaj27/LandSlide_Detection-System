Landslide Detection with UNet Model

This repository contains the code and resources for a deep learning-based landslide detection project using the UNet architecture. The model processes satellite imagery and generates segmentation masks for landslide areas.

Project Overview

This project leverages satellite imagery with RGB, NDVI (Normalized Difference Vegetation Index), DEM (Digital Elevation Model), and slope data to detect landslides. The UNet model is trained on preprocessed data to produce segmentation masks, highlighting potential landslide regions.

Features

Processes 128x128 satellite images.

Custom preprocessing pipeline including NDVI calculation, slope, and DEM normalization.

Implements a custom loss function (Dice Loss).

Tracks precision, recall, and F1-score during training.

Supports visualization of training, validation, and prediction results.

Dataset

The project uses the following datasets:

Training Data: Satellite images and corresponding segmentation masks.

Validation Data: Independent set of images for model evaluation.

Organize the dataset in the following structure:

/LandSlideDataset
  /TrainData
    /img
    /mask
  /ValidData
    /img
    /mask

Workflow

1. Data Preprocessing

Load image and mask datasets from .h5 files.

Calculate NDVI values.

Normalize RGB, slope, and elevation data.

Combine RGB, NDVI, slope, and elevation into a single tensor for training.

2. Model Definition

The UNet architecture is implemented with:

Contracting path for feature extraction.

Expanding path for upsampling and mask prediction.

The model uses:

Loss function: Binary Crossentropy with Dice Loss.

Metrics: Precision, recall, F1-score, and accuracy.

3. Training

Train the model using x_train and y_train with an 80/20 train-validation split.

Save the best model using a checkpoint callback.

4. Evaluation

Evaluate the model on validation data.

Generate prediction masks and visualize results.

5. Prediction

Use the trained model to predict landslide regions on new validation data.

Save generated masks to the output directory.

Usage

Training the Model

Run the following script to train the model:

python train.py

Visualizing Results

Modify the index in the script to visualize specific images or masks:

img = <index>

Prediction

Generate predictions for the validation dataset:

python predict.py

Key Functions

Data Preprocessing

calculate_ndvi: Computes NDVI from NIR and RED channels.

normalize_data: Normalizes RGB, slope, and elevation values.

Metrics

precision_m: Precision calculation.

recall_m: Recall calculation.

f1_m: F1-score calculation.

Model

unet_model: Defines the UNet architecture with convolutional and transposed convolutional layers.

Results

Loss: 0.036

Accuracy: 98.7%

F1-Score: 0.73

Precision: 0.80

Recall: 0.67

Visualizations

Training and validation loss.

Precision, recall, and F1-score curves.

Predicted masks vs. ground truth.

Directory Structure

.
|-- train.py
|-- predict.py
|-- model_save.h5
|-- best_model.h5
|-- /LandSlideDataset
    |-- /TrainData
    |   |-- /img
    |   |-- /mask
    |-- /ValidData
        |-- /img
        |-- /mask

Acknowledgments

Satellite image data preprocessing inspired by remote sensing methodologies.

UNet architecture adapted from segmentation literature.
