import configparser
import logging
from datetime import datetime

import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.layers import Flatten, Dropout
from keras.layers import Input, Dense, Conv2D, MaxPool2D
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.python.keras.callbacks import TensorBoard

from mnist_data_helpers import extract_data
from mnist_data_helpers import extract_labels

tf.get_logger().setLevel(logging.ERROR)


def mnist_train():
    print("------------------------------------ MNIST TRAINING ------------------------------------\n")

    # Params for MNIST
    validation_size = 5000  # Size of the validation set.

    # Config file parser
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Loading Fashion MNIST data
    print('Loading data...\n')

    # Data paths.
    train_data_filename = './data/images/train-images-idx3-ubyte.gz'
    train_labels_filename = './data/images/train-labels-idx1-ubyte.gz'
    test_data_filename = './data/images/t10k-images-idx3-ubyte.gz'
    test_labels_filename = './data/images/t10k-labels-idx1-ubyte.gz'

    # Extract it into numpy arrays.
    X_train = extract_data(train_data_filename, 60000)
    X_test = extract_data(test_data_filename, 10000)
    y_train = extract_labels(train_labels_filename, 60000)
    y_test = extract_labels(test_labels_filename, 10000)

    """
    # Generate a validation set.
    validation_data = train_data[:validation_size, :]
    validation_labels = train_labels[:validation_size, :]
    train_data = train_data[validation_size:, :]
    train_labels = train_labels[validation_size:, :]
    """

    # initialize the label names
    categories = ["top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]

    # MNIST parameters
    width = 28
    height = 28
    depth = 1

    # Set parameters
    pool_sizes = [2, 3, 4, 5]
    default_pool_size = pool_sizes[0]
    num_filters = 64

    learning_rate = float(config['CONFIGURATION']['Learning_Rate'])
    dropout_value = float(config['CONFIGURATION']['Dropout'])

    epochs = int(config['CONFIGURATION']['Training_Iterations'])
    filter_sizes = config.get('CONFIGURATION', 'Filter_Size').split()  # [5, 3, 8] #changed for now from 3,5,8
    batch_sizes = config.get('CONFIGURATION', 'Batch_Size').split()  # [1, 4, 16, 64]
    num_layers = config.get('CONFIGURATION', 'Tot_Layers').split()  # [4, 6, 8, 10]

    # Use these below to train just one model with chosen batch size and number of layers
    # batch_sizes = [1]
    # num_layers = [4]

    X_train = X_train.reshape(X_train.shape[0], height, width, 1)
    X_test = X_test.reshape(X_test.shape[0], height, width, 1)
    # input_shape = (height, width, 1)

    """ Models creation """
    for k in range(1, len(filter_sizes)):
        for j in range(0, len(batch_sizes)):
            for i in range(0, len(num_layers)):
                print("")
                print("_______________________________________________________________________________________________")
                print("")
                print("\nCreating Model...")
                batch_size = int(batch_sizes[j])
                layers = int(num_layers[i])
                conv_size = int(filter_sizes[k])
                print("Model with:")
                print("    batch size: " + str(batch_size))
                print("    number of layers: " + str(layers))
                print("    convolution size: " + str(conv_size))
                # Input layers
                inputs = Input(shape=(height, width, depth,), dtype='float32')

                # Different number of hidden layers
                if layers == 4:  # (conv + pool + 2 * fc)
                    conv_0 = Conv2D(num_filters, kernel_size=(conv_size, width), padding='valid',
                                    kernel_initializer='normal', activation='relu')(inputs)
                    maxpool_0 = MaxPool2D(pool_size=(height - conv_size + 1, 1), strides=(1, 1),
                                          padding='valid')(conv_0)
                    flatten = Flatten()(maxpool_0)
                elif layers == 6:  # (2 * (conv + pool) + 2 * fc)
                    conv_0 = Conv2D(num_filters, kernel_size=conv_size, padding='valid',
                                    kernel_initializer='normal', activation='relu')(inputs)
                    maxpool_0 = MaxPool2D(pool_size=pool_sizes[0], strides=(1, 1), padding='valid')(conv_0)
                    conv_1 = Conv2D(num_filters, kernel_size=(conv_size, width - conv_size),
                                    padding='valid', kernel_initializer='normal', activation='relu')(maxpool_0)
                    maxpool_1 = MaxPool2D(pool_size=(height - 2 * conv_size + 1, 1), strides=(1, 1),
                                          padding='valid')(conv_1)
                    flatten = Flatten()(maxpool_1)
                elif layers == 8:  # (3 * (conv + pool) + 2 * fc)
                    conv_0 = Conv2D(num_filters, kernel_size=conv_size, padding='valid',
                                    kernel_initializer='normal', activation='relu')(inputs)
                    maxpool_0 = MaxPool2D(pool_size=pool_sizes[0], strides=(1, 1), padding='valid')(conv_0)
                    conv_1 = Conv2D(num_filters, kernel_size=conv_size, padding='valid',
                                    kernel_initializer='normal', activation='relu')(maxpool_0)
                    maxpool_1 = MaxPool2D(pool_size=default_pool_size, strides=(1, 1), padding='valid')(conv_1)
                    conv_2 = Conv2D(num_filters, kernel_size=(conv_size, width - 2 * conv_size),
                                    padding='valid', kernel_initializer='normal', activation='relu')(maxpool_1)
                    maxpool_2 = MaxPool2D(pool_size=(height - 3 * conv_size + 1, 1), strides=(1, 1),
                                          padding='valid')(conv_2)
                    flatten = Flatten()(maxpool_2)
                else:  # layers == 10 (4 * (conv + pool) + 2 * fc)
                    conv_0 = Conv2D(num_filters, kernel_size=conv_size, padding='valid',
                                    kernel_initializer='normal', activation='relu')(inputs)
                    maxpool_0 = MaxPool2D(pool_size=default_pool_size, strides=(1, 1), padding='valid')(conv_0)
                    conv_1 = Conv2D(num_filters, kernel_size=conv_size, padding='valid',
                                    kernel_initializer='normal', activation='relu')(maxpool_0)
                    maxpool_1 = MaxPool2D(pool_size=default_pool_size, strides=(1, 1), padding='valid')(conv_1)
                    conv_2 = Conv2D(num_filters, kernel_size=conv_size, padding='valid',
                                    kernel_initializer='normal', activation='relu')(maxpool_1)
                    maxpool_2 = MaxPool2D(pool_size=default_pool_size, strides=(1, 1), padding='valid')(conv_2)
                    conv_3 = Conv2D(num_filters, kernel_size=(conv_size, width - 3 * conv_size),
                                    padding='valid', kernel_initializer='normal', activation='relu')(maxpool_2)
                    maxpool_3 = MaxPool2D(pool_size=(height - 4 * conv_size + 1, 1), strides=(1, 1),
                                          padding='valid')(conv_3)
                    flatten = Flatten()(maxpool_3)

                # Output layers
                dense1 = Dense(units=num_filters, activation='relu')(flatten)
                dropout = Dropout(dropout_value)(dense1)
                output = Dense(units=len(y_train[0]), activation='softmax')(dropout)

                # TensorBoard for visualization
                tensorboard = TensorBoard(log_dir="./logs/{}".format(datetime.now()))

                # this creates a model that includes
                model = Model(inputs=inputs, outputs=output)

                # printing shape of the output tensor for each layer
                print("Model structure: \n")
                print("Tensor shape: " + str(model.layers[0].input_shape))
                for layer in model.layers:
                    print("---------------------------------- Layer: " + layer.name)
                    print("Tensor shape: " + str(layer.output_shape))
                print("")

                model.run_eagerly = True  # Otherwise TensorBoard doesn't work
                checkpoint = ModelCheckpoint(
                    './data/models/{}l-{}b-{}c-mnist_trained.hdf5'.format(layers, batch_size, conv_size),
                    monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
                adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
                model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

                """ Training phase """
                print("Training Model...")
                history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                                    callbacks=[checkpoint, tensorboard],
                                    validation_data=(X_test, y_test))  # starts training


mnist_train()
