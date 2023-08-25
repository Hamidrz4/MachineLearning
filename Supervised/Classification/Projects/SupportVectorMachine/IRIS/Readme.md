# SVM Kernel Comparison

This repository contains Python code that demonstrates the performance of Support Vector Machines (SVMs) with different kernel functions on various datasets. The `scikit-learn` library is used for SVM implementation, and the `mlxtend` library is used to visualize the decision regions.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Results](#results)
- [Dependencies](#dependencies)
- [License](#license)

## Introduction
Support Vector Machines are powerful classifiers that can separate data points using different kernel functions. This repository showcases the performance of SVMs with linear, radial basis function (RBF), and polynomial kernels on various datasets. Decision regions are visualized using the `mlxtend` library.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/svm-kernel-comparison.git
   cd svm-kernel-comparison
   ```

2. Install the required libraries:
   ```sh
   pip install numpy matplotlib scikit-learn mlxtend
   ```

## Usage
1. Open the `svm_kernel_comparison.ipynb` notebook or run the provided Python script (`svm_kernel_comparison.py`).

2. The code performs the following steps for each dataset:
   - Loads the required datasets using `make_moons`, `make_circles`, and custom data generation.
   - Applies SVM with linear, RBF, and polynomial kernels to each dataset.
   - Visualizes the decision regions for each SVM using subplots and `plot_decision_regions` from `mlxtend`.

## Datasets
The code uses the following datasets:
- Iris dataset (2 features)
- make_moons dataset
- make_circles dataset
- XOR dataset (custom generated)

## Results
The code demonstrates the decision regions of SVMs with different kernel functions on various datasets. It visually compares how SVMs with different kernels perform in separating data points.

## Dependencies
- numpy
- matplotlib
- scikit-learn
- mlxtend
