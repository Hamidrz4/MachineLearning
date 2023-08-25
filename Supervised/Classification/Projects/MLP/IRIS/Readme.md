# Perceptron Classification on Iris Dataset

This repository contains code for performing classification using the Perceptron algorithm on the famous Iris dataset. The Perceptron is a simple binary classification algorithm that learns a linear decision boundary to separate data points belonging to different classes.

## Table of Contents

- [Introduction](#introduction)
- [Code Overview](#code-overview)
  - [Importing Libraries](#importing-libraries)
  - [Loading and Visualizing Data](#loading-and-visualizing-data)
  - [Preparing the Training Set](#preparing-the-training-set)
  - [Training the Perceptron](#training-the-perceptron)
  - [Plotting Decision Boundary](#plotting-decision-boundary)
  - [Cross-Validation](#cross-validation)
  - [Plotting Misclassifications](#plotting-misclassifications)
- [Results](#results)

## Introduction

The Perceptron algorithm is a building block for more complex neural networks. In this project, we demonstrate how to use the Perceptron algorithm to classify Iris flowers into different species based on their sepal and petal dimensions.

## Code Overview

### Importing Libraries

Import the necessary libraries and modules for data manipulation, visualization, and machine learning.

### Loading and Visualizing Data

Load the Iris dataset, create a DataFrame, and visualize the relationships between feature pairs using seaborn's pairplot.

### Preparing the Training Set

Select a subset of the dataset for training. Extract features and the target variable and convert them to numpy arrays.

### Training the Perceptron

Create a Perceptron model, specify hyperparameters, and fit the model to the training data.

### Plotting Decision Boundary

Use mlxtend's plot_decision_regions function to visualize the decision boundary of the trained Perceptron model.

### Cross-Validation

Perform cross-validation using scikit-learn's cross_val_score with the Leave-One-Out strategy.

### Plotting Misclassifications

Plot the number of misclassifications at each iteration of the Perceptron training.

## Results

The Perceptron demonstrates its ability to classify Iris flowers effectively. The decision boundary provides a clear separation between different species.
