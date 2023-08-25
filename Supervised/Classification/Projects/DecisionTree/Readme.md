# Decision Tree Classifier for Tumor Classification

This repository contains code for tumor classification using a Decision Tree Classifier. Decision Trees are versatile machine learning algorithms used for both classification and regression tasks.

## Table of Contents

- [Introduction](#introduction)
- [Code Overview](#code-overview)
  - [Importing Libraries](#importing-libraries)
  - [Loading and Preprocessing Data](#loading-and-preprocessing-data)
  - [Training the Decision Tree Model](#training-the-decision-tree-model)
  - [Evaluating Model Performance](#evaluating-model-performance)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Evaluating Tuned Model](#evaluating-tuned-model)
  - [Visualizing the Decision Tree](#visualizing-the-decision-tree)
- [Results](#results)

## Introduction

Decision Trees are powerful algorithms that make decisions by repeatedly partitioning the feature space into subsets based on the most discriminative features. In this project, we use a Decision Tree to classify tumors into different classes based on their features.

## Code Overview

### Importing Libraries

Import the necessary libraries and modules for data manipulation, visualization, and machine learning.

### Loading and Preprocessing Data

Load the tumor dataset, split it into features (x) and target (y), and perform data preprocessing.

### Training the Decision Tree Model

Create a Decision Tree model with specified hyperparameters and fit it to the training data.

### Evaluating Model Performance

Evaluate the Decision Tree model's performance on both the training and test datasets using classification reports.

### Hyperparameter Tuning

Perform a grid search to find the best hyperparameters for the Decision Tree model using cross-validation.

### Evaluating Tuned Model

Evaluate the tuned Decision Tree model's performance on both the training and test datasets using classification reports.

### Visualizing the Decision Tree

Visualize the trained Decision Tree using the `plot_tree` function.

## Results

The Decision Tree model demonstrates its ability to classify tumors with high accuracy. Hyperparameter tuning further improves the model's performance, resulting in a more balanced and accurate classification. The visualization of the Decision Tree provides insights into the decision-making process of the model.
