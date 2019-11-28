import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import optimizers


sequence_length = 100


def create_network():
    """ create the structure of the neural network """
    l_rate = .0001
    model = Sequential()
    model.add(Dense(100, input_shape=(sequence_length,), kernel_initializer='random_normal', activation='relu'))
    model.add(Dense(60, kernel_initializer='random_normal', activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(60, kernel_initializer='random_normal', activation='relu'))
    # model.add(BatchNorm())
    model.add(Dense(45, kernel_initializer='random_normal', activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(20, kernel_initializer='random_normal', activation='relu'))
    model.add(Dense(2, kernel_initializer='random_normal', activation='softmax'))
    adam = optimizers.Adam(lr=l_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['categorical_accuracy'])
    model.load_weights('../ClassifierNet/classification-improvement-200.hdf5')

    return model


def prediction(model, network_input):
    print(network_input.shape)
    label = np.argmax(model.predict(network_input))
    if label == 1:
        category = 'light'
    else:
        category = 'dark'
    return category


if __name__ == '__main__':
    classifier_network = create_network()
