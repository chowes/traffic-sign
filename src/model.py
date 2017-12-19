from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import SGD
from keras import backend as K

# specify image data format - needed for the preprocessing we do below
K.set_image_data_format('channels_first')


class CNNModel:
    model = None

    def __init__(self, image_sz=48, num_classes=43, learning_rate=.001, batch_size=32, epochs=400):
        """
        Initialize our model - this configuration seems to be pretty "standard", as far as I can tell.
        I tried to fiddle with the convolutional layer parameters but it just breaks (and I have no idea why).
        """
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.epochs=epochs

        # each layer feeds into the next layer - a bit different than the Sermanet paper
        self.model = Sequential()

        # convolutional layers followed by pooling layers
        self.model.add(Conv2D(32, (3, 3), padding='same', input_shape=(3, image_sz, image_sz), activation='relu'))
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

        self.model.add(Dense(num_classes, activation='softmax'))


    def train(self, X, Y):
        # use stochastic gradient descent as our optimizer
        sgd = SGD(lr=self.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)

        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        history = self.model.fit(X, Y,
                      batch_size=self.batch_size,
                      epochs=self.epochs,
                      validation_split=0.2,
                      callbacks=[LearningRateScheduler(lambda epoch: self.learning_rate * (0.1 ** int(epoch/10))),
                                 ModelCheckpoint('model.h5', save_best_only=True)]
                      )

        return history


    def predict(self, X):
        return self.model.predict_classes(X)

    def evaluate(self, X, Y):
        return self.model.evaluate(x=X, y=Y, verbose=0)
