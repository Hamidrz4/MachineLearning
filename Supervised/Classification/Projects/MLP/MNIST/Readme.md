# Handwritten Digit Recognition using MLP Classifier

This repository contains code for training and evaluating a Multi-Layer Perceptron (MLP) Classifier model for handwritten digit recognition using the MNIST dataset.

## Getting Started

These instructions will help you set up and run the code on your local machine.

### Prerequisites

Before running the code, make sure you have the following libraries installed:

- numpy
- pandas
- scikit-learn
- matplotlib
- mlxtend

You can install them using the following command:

```bash
pip install numpy pandas scikit-learn matplotlib mlxtend
```

### Dataset

The MNIST dataset is used for training and testing the model. Download the dataset and save the `mnist_train.csv` and `mnist_test.csv` files in the same directory as the code.

### Code Structure

The code is structured as follows:

- **Import Libraries**: Import necessary libraries for the project.

- **Load and Preprocess Data**: Load the dataset, separate labels and features, and normalize the pixel values to the range [0, 1].

- **Train the Model**: Train the MLP Classifier using the training data. The model has one hidden layer with 100 units, stochastic gradient descent (SGD) optimizer, and a maximum of 100 iterations. The training process includes monitoring the loss and stopping when it doesn't improve significantly.

- **Evaluate the Model**: Evaluate the trained model using the test dataset. Generate classification reports, including precision, recall, and F1-score, and display the confusion matrix.

- **Cross-Validation**: Perform 10-fold cross-validation to assess the model's performance and check for overfitting.

- **Comparison of Validation and Test Scores**: Compare the highest validation score achieved during cross-validation with the test score.

- **Confusion Matrix Visualization**: Display a visual representation of the confusion matrix using the mlxtend library.

## Results

The model achieves an accuracy of approximately 98% on the test dataset. Cross-validation results also show consistent performance with an average score of around 97.5%.

## Conclusion

The MLP Classifier demonstrates strong performance in recognizing handwritten digits from the MNIST dataset. The model achieves high accuracy on both validation and test datasets, suggesting its effectiveness in real-world applications.

Feel free to experiment with hyperparameters, architectures, and optimization techniques to further improve the model's performance.
```

Remember that this is just a sample README and you can modify it to match your project's details and structure.
