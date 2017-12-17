from sklearn.utils import shuffle
from skimage import exposure
import warnings 
import pickle
import csv


def read_data(filename, col_names):
    """
    Read data from file. I found a pre-marshalled version of the GTSRB data which makes importing it a lot easier...
    """

    # pickle files have to be read in binary mode
    with open(filename, "rb") as f:
        data = pickle.load(f)

    # use map to create a tuple of our data set corresponding with the given column names ("features", "labels")
    dataset = tuple(map(lambda i: data[i], col_names))

    return dataset


def preprocess_img_data(x, num_classes):


    

