#!/usr/bin/python3

import sys
import time
import os
import numpy as np

from image_processing import import_training_set, import_test_set, preprocess_image
from perf_plots import plot_loss, plot_accuracy
from model import TrafficModel


def main():
    
    train_x, train_y = import_training_set('../images/Final_Training/Images/')

    print("Starting training, this is a good time to go get a coffee...\n")

    traffic_model = TrafficModel(image_sz=48, num_classes=43)
    
    start_time = time.time()
    traffic_model.set_model_params(epochs=30)
    history = traffic_model.train(train_x, train_y)
    end_time = time.time()

    print(end_time - start_time)

    print("Evaluating performance on our test set...")

    test_x, test_y = import_test_set("../images/Final_Test/")
    print(traffic_model.test(test_x, test_y))


if __name__ == '__main__':
    main()