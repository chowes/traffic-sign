#!/usr/bin/python3

import sys
import numpy as np

from image_processing import read_data, convert_to_greyscale
from log import show_time_elapsed, show_progress

def main():

    print("Read training and test data...\n")
    
    train_x, train_y = read_data("../images/train.p", ['features', 'labels'])
    test_x, test_y = read_data("../images/test.p", ['features', 'labels'])

    print("Training images: " + str(train_y.shape[0]))
    print("Test images: " + str(test_y.shape[0]))
    print("Classes: " + str(np.unique(test_y).shape[0]) + "\n")

    print("Perfoming image preprocessing...\n")

    train_x = convert_to_greyscale(train_x)
    test_x = convert_to_greyscale(test_x)

    print("Starting training, this is a good time to go get a coffee...")


    print("Evaluating performance on our test set...")




if __name__ == '__main__':
    main()