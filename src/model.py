from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras import backend as K

import numpy as np

from sklearn.model_selection import train_test_split

# specify image data format - needed for the preprocessing we do below
K.set_image_data_format('channels_first')


class TrafficModel:
    model = None

    def __init__(self, image_sz=48, num_classes=43):
        """
        Create a new model, set image size and number of classes here
        """

        self.image_sz = image_sz
        self.num_classes = num_classes
        


    def load_model(self, file_path):
        """
        Load a pretrained model from file
        """

    def set_model_params(self, learning_rate=.001, batch_size=32, epochs=30):
        """
        Setup model paramters for training
        """

        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.epochs=epochs

        # each layer feeds into the next layer - a bit different than the Sermanet paper
        self.model = Sequential()

        # convolutional layers followed by pooling layers
        self.model.add(Conv2D(32, (3, 3), padding='same', input_shape=(3, self.image_sz, self.image_sz), activation='relu'))
        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        # classifier
        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(self.num_classes, activation='softmax'))


    def train(self, x_train, y_train):
        # use stochastic gradient descent as our optimizer
        sgd = SGD(lr=self.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)

        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.5)
        # print(x_train.shape)
        # print(x_test.shape)

        # train_datagen = ImageDataGenerator(
        #     featurewise_center=True,
        #     featurewise_std_normalization=True,
        #     zca_whitening=True,
        #     rotation_range=20,
        #     width_shift_range=0.2,
        #     height_shift_range=0.2)

        # test_datagen = ImageDataGenerator(
        #     featurewise_center=True,
        #     featurewise_std_normalization=True,
        #     zca_whitening=True,
        #     rotation_range=20,
        #     width_shift_range=0.2,
        #     height_shift_range=0.2)

        # train_datagen.fit(x_train)
        # test_datagen.fit(x_test)

        # history = self.model.fit_generator(
        #     generator=train_datagen.flow(x_train, y_train, batch_size=self.batch_size),
        #     validation_data=test_datagen.flow(x_test, y_test, batch_size=self.batch_size),
        #     epochs=self.epochs,
        #     # steps_per_epoch=len(x_train),
        #     callbacks=[LearningRateScheduler(lambda epoch: self.learning_rate * (0.1 ** int(epoch/10))),
        #                ModelCheckpoint('model.h5', save_best_only=True)])

        history = self.model.fit(x_train, y_train,
                      batch_size=self.batch_size,
                      epochs=self.epochs,
                      validation_split=0.2,
                      callbacks=[LearningRateScheduler(lambda epoch: self.learning_rate * (0.1 ** int(epoch/10))),
                                 ModelCheckpoint('model.h5', save_best_only=True)])

        return history


    def test(self, test_x, test_y):
        """
        Test the model on a test set
        """

        predictions = self.model.predict_classes(test_x)

        return np.sum(predictions == test_y) / predictions.shape[0]


    def predict(self, X):
        """
        Predict label for a given example
        """
        return self.model.predict_classes(X)

