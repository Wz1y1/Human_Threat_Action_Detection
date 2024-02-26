# Real-Time Threat Action Detection System

## Overview
This project implements a real-time action detection system designed for drone-based monitoring. Leveraging a blend of 1D Convolutional Neural Networks (CNNs) and Recurrent Neural Network (RNN) architectures, including Bidirectional Long Short-Term Memory (Bi-LSTM) 
and Bidirectional Gated Recurrent Units (Bi-GRU), also combined with MediaPipe pre-trained model for efficient landmark detection from video feeds. The system is optimized for low latency and high accuracy in diverse operational environments.

## Features
- **Real-Time Detection:** Fast and efficient detection of actions in real-time video streams.
- **MediaPipe Integration:** Incorporates MediaPipe for robust landmark detection from raw video frames.
- **Low Latency:** Designed to minimize response time for timely action recognition by muti-thread architecture

## Training Data and Customization

This system is designed to be adaptive and customizable, supporting the incorporation of user-collected data to train models for detecting specific threat actions unique to different operational environments.

### Data Collection Algorithm

- **Overview**: Incorporates data collection algorithm that facilitates the gathering of training data in real-world scenarios.
- **User Participation**: Allows users to contribute by recording and submitting video sequences of potential threat actions, enhancing the model's ability to recognize a wide range of activities.


## Model Architecture Overview

This real-time action detection system is optimized for drone-based surveillance, combining MediaPipe with custom models for facial expression and body posture analysis to achieve high accuracy and low latency.

### Data Preprocessing with MediaPipe

- **Description**: Utilizes MediaPipe for extracting key landmarks from video frames, providing a clean dataset for model inputs.

### Facial Expression Model

- **Functionality**: Identifies specific facial expressions from preprocessed landmarks, trained on a diverse dataset to recognize a wide range of emotions.
- **Architecture**: Employs a CNN designed to process spatial relationships between facial landmarks for accurate expression identification and Bi-GRU for sequential analysis.


### Body Posture Model

- **Functionality**: Analyzes body movements and positions, identifying actions like punching or waving.
- **Architecture**: Using three Bi-LSTM sub-model to capture spatial features and temporal dynamics of body movements.



