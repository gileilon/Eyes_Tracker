# Eyes_Tracker

## Overview

The Eye Tracker project is a Python-based system designed for eye tracking. The primary goal is to create a comprehensive solution that involves presenting dots on the screen, capturing frames from the computer camera, and utilizing a Support Vector Machine (SVM) classifier from scikit-learn to determine the direction of the eyes.

## Project Structure

1. **DataCreator**
   - The DataCreator class is responsible for generating or collecting data used in training and testing the eye tracker.

2. **TrackingPointsTask**
   - The TrackingPointsTask class represents a task involving the presentation of dots on the screen. Frames are captured from the computer camera to create labeled training data by capturing images of eyes looking at various locations of the dots.

3. **DataConvertorForTrainingClassifier**
   - The DataConvertorForTrainingClassifier class converts data to a suitable format for training the SVM classifier.

4. **TrainingEyesTrackerManager**
   - This class acts as a manager, coordinating different project components such as data conversion, data creation, and training.

5. **Train_SVMs**
   - Train_SVMs is a class that uses scikit-learn's SVM classifier to train and save models for eye direction classification.

## Prerequisites

Ensure you have the following prerequisites installed before running the project:

- Python 3.x
- Required Python packages:
  - `numpy`
  - `PIL` (Pillow)
  - `os`
  - `time`
  - `cv2` (OpenCV)
  - `pygame`
  - `scikit-learn` (sklearn)
  - `pandas`

