import numpy as np
import warnings 
import pickle
import csv


def read_data(filename, col_names):
    """
    Read data from file. I found a pre-marshalled version of the GTSRB data which makes importing it a lot easier.
    In order to read use this data in a meaningful way we extract "features" and "labels". 

    We return these as a tuple.
    """

    # Pickle files have to be read in binary mode!
    with open(filename, "rb") as f:
        data = pickle.load(f)

    # Use map to create a tuple of our data set corresponding with the given column names ("features", "labels")
    dataset = tuple(map(lambda i: data[i], col_names))

    return dataset


def convert_to_greyscale(img_data):
    """
    Convert image data to a single greyscale channel and scale down to within range [0, 1]
    """

    # These weights are (apparently) the accepted way to convert to grey scale as per PAL and NTSC (and Wikipedia)
    img_data = (0.299 * img_data[:, :, :, 0] + 0.587 * img_data[:, :, :, 1] + 0.114 * img_data[:, :, :, 2]) / 255
    
    # Scale all features to be in [0, 1]
    img_data = (img_data / 255)
    img_data = img_data.astype(np.float32)

    # As per the Sermanet paper, we train only on a single grey channel
    img_data = img_data.reshape(img_data.shape + (1,))

    return img_data
