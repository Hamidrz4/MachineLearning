# SVM Decision Boundary Visualization

This repository contains Python code that demonstrates the visualization of SVM decision boundaries using different kernels and datasets. The code utilizes the `mlxtend` library to plot decision regions of the SVM models.

## Introduction

Support Vector Machines (SVMs) are powerful machine learning models used for classification and regression tasks. This repository showcases the use of SVMs with various kernels on different datasets to visualize their decision boundaries.

## Usage

The provided Python script performs the following steps:
- Imports necessary libraries for data manipulation and visualization.
- Loads sample datasets: Iris, make-moons, make-circles, and XOR.
- For each dataset, applies SVM with different kernels (linear, radial basis function, and polynomial).
- Uses `plot_decision_regions` to visualize decision boundaries on subplots for each SVM model.

## Datasets

1. **Iris Dataset:** A well-known dataset with features representing iris flower characteristics. SVM is used to classify different iris species.
2. **make-moons Dataset:** Synthetic dataset containing two interleaving half-moons. SVM is applied to separate the two classes.
3. **make-circles Dataset:** Synthetic dataset with classes arranged in concentric circles. SVM is used to classify the circular regions.
4. **XOR Dataset:** A custom dataset where classes are defined by the XOR operation. SVMs are used to learn the XOR pattern.

## Dependencies

- numpy
- matplotlib
- sklearn
- mlxtend

## Results

The code generates subplots for each dataset and kernel combination, showing the decision boundary and the classification regions created by the SVM. The visualizations help in understanding how different kernels handle different datasets and types of decision boundaries.
