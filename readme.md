# Pedestrian Detection System

This repository contains code for a Pedestrian Detection System that utilizes a machine learning model trained on a pedestrian dataset and incorporates the technique of moving windows. The system is designed to detect pedestrians in images by employing a sliding window approach. This technique involves scanning the image with a window of fixed size and moving it across the image in a systematic manner. At each position, the machine learning model is applied to determine whether the window contains a pedestrian or not. By iterating this process over different scales and positions, the system can effectively identify pedestrians in various parts of the image. The combination of the trained model and the moving windows technique enables accurate pedestrian detection in a wide range of scenarios.

## Requirements

The code requires the following dependencies:

- Python 3.7 or above
- OpenCV (cv2)
- scikit-image
- scikit-learn
- joblib
- PyYAML
- matplotlib

## Code Structure

1. pedestrian_dataset_loader.py: Defines a class PedestrianDatasetLoader for loading and preprocessing the pedestrian dataset.

2. model_cfg.yaml: YAML file containing the configuration for the pedestrian detection model.

3. classifier_cfg.yaml: YAML file containing the configuration for the MLP classifier used in training.

4. train_classifier.py: Script for training the pedestrian detection model.

5. test_classifier.py: Script for testing the trained pedestrian detection model.

6. dataset: images that contain people and images that don't contain people. Images are used for
   training and tresting the model.
