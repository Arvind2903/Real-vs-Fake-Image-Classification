# Fake-vs-Real-Image-Classification

## Introduction

This project focuses on classifying images into two categories: "fake" and "real". The goal is to distinguish between manipulated or fake images and authentic ones using deep learning techniques and transfer learning.

## Dataset

The dataset consists of images categorized into two classes: "fake" and "real". It is divided into training, validation, and test sets for model training, evaluation, and testing, respectively. 

### Data Preprocessing

- Data augmentation techniques such as rotation, rescaling, brightness adjustment, and horizontal/vertical flipping are applied to the training set to increase its size and diversity.
- Images are resized to 224x224 pixels to match the input size expected by the VGG16 model.
- Pixel values are normalized to the range [0, 1] to facilitate model convergence.

## Model Architecture

- Transfer learning with the VGG16 pre-trained convolutional neural network is employed as the base model.
- The fully connected layers of VGG16 are replaced with two additional Dense layers with ReLU activation.
- The output layer consists of a single neuron with sigmoid activation, yielding binary classification probabilities.

## Training

- The model is trained using the Adam optimizer with a learning rate of 0.001.
- Class weights are computed to handle class imbalance in the training data.
- Model training is monitored using early stopping based on validation loss to prevent overfitting.

## Evaluation

- Model performance is evaluated on the test set using accuracy, precision, recall, and F1-score metrics.
- Confusion matrix visualization is employed to assess classification performance and identify any misclassifications.

## Results

- The VGG16-based model achieves promising results in classifying fake and real images.
- Performance metrics such as accuracy, precision, recall, and F1-score demonstrate the effectiveness of the model.
- The confusion matrix provides insights into the model's classification behavior and error patterns.

## Conclusion

- The project demonstrates the potential of deep learning approaches for detecting manipulated images and preserving authenticity.
- Further improvements and optimizations can be explored to enhance model performance and robustness.
