# MNIST Digit Classification using MLP Classifier

This repository contains code for training and evaluating an MLP Classifier model on the MNIST dataset for digit classification. The model is trained using the `sklearn` library's `MLPClassifier`.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Dependencies](#dependencies)
- [License](#license)

## Introduction
This project demonstrates the use of an MLP Classifier to classify handwritten digits from the MNIST dataset. The model is trained on the training set and evaluated on both the validation set and the test set.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/mnist-digit-classification.git
   cd mnist-digit-classification
   ```

2. Install the required libraries:
   ```sh
   pip install numpy pandas scikit-learn matplotlib mlxtend seaborn
   ```

3. Download the MNIST dataset files (`mnist_train.csv` and `mnist_test.csv`) and place them in the project directory.

## Usage
1. Open the `mnist_classification.ipynb` notebook or run the provided Python script (`mnist_classification.py`).

2. The code performs the following steps:
   - Imports the necessary libraries.
   - Loads the MNIST dataset.
   - Preprocesses the data by scaling pixel values.
   - Creates an MLP Classifier model and trains it on the training data.
   - Evaluates the model's performance on the test data and displays the classification report.
   - Performs k-fold cross-validation and reports validation scores.
   - Compares the maximum validation score with the test score.
   - Plots the confusion matrix for the test predictions.

## Results
The MLP Classifier achieves an accuracy of approximately 98.08% on the test set. The validation scores from k-fold cross-validation range from 97.27% to 98.17%, with a maximum score of 98.17%.

## Dependencies
- numpy
- pandas
- scikit-learn
- matplotlib
- mlxtend
- seaborn
