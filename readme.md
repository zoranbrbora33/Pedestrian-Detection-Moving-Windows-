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

1. pedestrian_dataset_loader.py:
   The file provides a class that encapsulates the functionality to load and preprocess a pedestrian dataset, making it easier to work with the data for training a pedestrian detection model. The dataset is split into training and testing sets, and information about the dataset is printed for analysis.

2. train_pedestrian_classifier.py:
   This script is meant to train a pedestrian classification model using an MLP classifier, visualize the training progress, and save the trained model and its configuration for future use.

3. classifier_inference.py:
   This script demonstrates pedestrian detection using a sliding window approach and a trained pedestrian classification model. The code iterates over different window positions, extracts regions of interest, computes HOG features, and predicts the probability of each ROI containing a pedestrian. If a sufficiently high probability is found, a bounding box is drawn around the detected pedestrian in the final image.

4. classifier_inference2.py:
   This script demonstrates pedestrian detection using a sliding window approach, followed by NMS to remove redundant detections. The NMS algorithm helps eliminate overlapping bounding boxes and provides a more accurate representation of the detected pedestrians in the final image.

5. nms.py:
   This file contains code for performing non-maximum suppression (NMS) on a list of object detections. Non-maximum suppression is a technique commonly used in object detection tasks to eliminate duplicate or overlapping detections and retain only the most confident and accurate ones.

6. dataset: images that contain people and images that don't contain people. Images are used for
   training and testing the model.

7. model_cfg.yaml: YAML file containing the configuration for the pedestrian detection model.

8. classifier_cfg.yaml: YAML file containing the configuration for the MLP classifier used in training.
