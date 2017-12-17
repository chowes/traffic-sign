#!/usr/bin/python3

import sys
import numpy as np
from image_processing import read_data

def main():

    print("Read training and test data...")
    
    train_x, train_y = read_data("../images/train.p", ['features', 'labels'])
    test_x, test_y = read_data("../images/test.p", ['features', 'labels'])
    print("Loaded " + str(train_y.shape[0]) + " training images, " + str(test_y.shape[0]) + " test images from " + str(np.unique(test_y).shape[0]) + " classes")

    




if __name__ == '__main__':
    main()