import os
import glob
import h5py
import random
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import io, color, exposure, transform
from keras.utils import np_utils


def import_training_set(image_dir='../images/Final_Training/Images/'):
    """
    Read images from GTSRB file, uses kind of a hacky fix to get class names. The idea comes from:
    https://chsasank.github.io/keras-tutorial.html
    """

    images = []
    labels = []

    # get all of our image paths
    all_paths = glob.glob(os.path.join(image_dir, '*/*.ppm'))

    # we have to shuffle here since x and y indices need to match
    np.random.shuffle(all_paths)

    for image_path in all_paths:
        image = preprocess_image(io.imread(image_path))
        label = int(image_path.split('/')[-2])
        images.append(image)
        labels.append(label)

    # we need x to be a matrix of 32 bit floats (defined in numpy)
    x_train = np.array(images, dtype='float32')

    # we have to use one-hot encoding
    y_train = np_utils.to_categorical(labels, np.unique(labels).shape[0])

    return x_train, y_train


def import_test_set(image_dir='../images/Final_Training/'):
    test = pd.read_csv(os.path.join(image_dir, 'GT-final_test.csv'), sep=';')
    image_dir = os.path.join(image_dir, "Images")

    test_x = []
    test_y = []

    for file_name, class_id in zip(list(test['Filename']), list(test['ClassId'])):
        image_path = os.path.join(image_dir, file_name)
        test_x.append(preprocess_image(io.imread(image_path)))
        test_y.append(class_id)

    test_x = np.array(test_x)
    test_y = np.array(test_y)

    return test_x, test_y


def preprocess_image(image, image_sz=48):
    """
    Preprocess an image. Most of this is stuff that needs to be done for the Keras CNN model to work,
    as recommended by: https://chsasank.github.io/keras-tutorial.html
    """

    # we need to convert to saturation, and value (HSV) coordinates
    hsv_image = color.rgb2hsv(image)
    hsv_image[:, :, 2] = exposure.equalize_hist(hsv_image[:, :, 2])
    image = color.hsv2rgb(hsv_image)

    # we have to crop to central square
    min_side = min(image.shape[:-1])
    centre = image.shape[0] // 2, image.shape[1] // 2
    image = image[centre[0] - min_side // 2:centre[0] + min_side // 2, centre[1] - min_side // 2:centre[1] + min_side // 2, :]

    # our model _needs_ images that are all the same size
    image = transform.resize(image, (image_sz, image_sz))

    # change colour axis
    image = np.rollaxis(image, -1)

    return image
