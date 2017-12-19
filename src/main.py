#!/usr/bin/python3

import sys
import os
import numpy as np
from image_processing import import_training_set, import_test_set, preprocess_image

from perf_plots import plot_loss, plot_accuracy
from model import CNNModel


def main():

    print("Read training and test data...\n")
    
    train_x, train_y = import_training_set('../images/Final_Training/Images/', 0.5)

    print("Training images: " + str(train_y.shape[0]))
    print("Classes: " + str(np.unique(train_y).shape[0]) + "\n")


    print("Starting training, this is a good time to go get a coffee...\n")

    traffic_model = CNNModel()
    history = traffic_model.train(train_x, train_y)

    print("Evaluating performance on our test set...")

    test_x, test_y = import_test_set("../images/Final_Test/")
    predictions = traffic_model.predict(test_x)
    test_accuracy = np.sum(predictions == test_y) / predictions.shape[0]

    print("Test accuracy: " + str(test_accuracy))

    plot_loss(history)
    plot_accuracy(history)


if __name__ == '__main__':
    main()