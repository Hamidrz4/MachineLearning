# SVM Decision Boundary Visualization

This repository contains Python code that demonstrates the visualization of an SVM decision boundary using matplotlib. The code preprocesses a dataset, applies an SVM model with a linear kernel, and plots the decision boundary on a meshgrid.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Results](#results)
- [Dependencies](#dependencies)

## Introduction
Support Vector Machines (SVMs) are powerful classifiers that can create decision boundaries to separate different classes of data. This repository showcases how to preprocess a dataset, train an SVM with a linear kernel, and visualize the decision boundary using matplotlib.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/svm-decision-boundary.git
   cd svm-decision-boundary
   ```

2. Install the required libraries:
   ```sh
   pip install pandas numpy matplotlib scikit-learn
   ```

## Usage
1. Open the provided Python script (`svm_decision_boundary.py`).

2. The code performs the following steps:
   - Reads a dataset from a CSV file.
   - Preprocesses the dataset, including feature transformation and normalization.
   - Trains an SVM with a linear kernel on the preprocessed data.
   - Generates a meshgrid and predicts the class labels for each point in the meshgrid.
   - Plots the original data points with different colors for different targets.
   - Plots the decision boundary on the meshgrid to visualize the classification regions.

## Dataset
The code uses a dataset (`age_salary.csv`) containing information about age, gender, and salary. It predicts whether an individual makes a purchase (`Purchased` column) based on the features.

## Results
The code demonstrates the visualization of an SVM decision boundary on a meshgrid. It plots the original data points and overlays the decision boundary to help visualize how the SVM classifies different regions.

## Dependencies
- pandas
- numpy
- matplotlib
- scikit-learn
