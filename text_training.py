import configparser
import logging
from datetime import datetime

import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import TensorBoard

from text_data_helpers import load_data

tf.get_logger().setLevel(logging.ERROR)


def text_train():
    print("------------------------------------ TEXT TRAINING ------------------------------------\n")

    # Config file parser
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Loading text data
    print('Loading data...\n')
    x, y, vocabulary, vocabulary_inv = load_data()
    # x.shape -> (10662, 56)
    # y.shape -> (10662, 2)
    # len(vocabulary) -> 18765
    # len(vocabulary_inv) -> 18765

    # Split test and train dataset (random_state is the seed used for random number generation)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # X_train.shape -> (8529, 56)
    # y_train.shape -> (8529, 2)
    # X_test.shape -> (2133, 56)
    # y_test.shape -> (2133, 2)

    # Set parameters
    sequence_length = x.shape[1]  # 56
    vocabulary_size = len(vocabulary_inv)  # 18765
    embedding_dim = 256
    pool_sizes = [2, 3, 4, 5]
    default_pool_size = pool_sizes[0]
    num_filters = 64

    learning_rate = float(config['CONFIGURATION']['Learning_Rate'])
    dropout_value = float(config['CONFIGURATION']['Dropout'])

    epochs = int(config['CONFIGURATION']['Training_Iterations'])
    filter_sizes = config.get('CONFIGURATION', 'Filter_Size').split()  # [3, 5, 8]
    batch_sizes = config.get('CONFIGURATION', 'Batch_Size').split()  # [1, 4, 16, 64]
    num_layers = config.get('CONFIGURATION', 'Tot_Layers').split()  # [4, 6, 8, 10]

    # Use these below to train just one model with chosen batch size and number of layers
    # batch_sizes = [1]
    # num_layers = [4]

    """ Models creation """
    for k in range(0, len(filter_sizes)):
        for j in range(0, len(batch_sizes)):
            for i in range(0, len(num_layers)):

                print("Creating Model...")
                batch_size = int(batch_sizes[j])
                layers = int(num_layers[i])
                conv_size = int(filter_sizes[k])
                print("Model with:")
                print("    batch size: " + str(batch_size))
                print("    number of layers: " + str(layers))
                print("    convolution size: " + str(conv_size))

                # Input layers
                inputs = Input(shape=(sequence_length,), dtype='int32')
                embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim,
                                      input_length=sequence_length)(inputs)
                reshape = Reshape((sequence_length, embedding_dim, 1))(embedding)

                # Different number of hidden layers
                if layers == 4:  # (conv + pool + 2 * fc)
                    conv_0 = Conv2D(num_filters, kernel_size=(conv_size, embedding_dim), padding='valid',
                                    kernel_initializer='normal', activation='relu')(reshape)
                    maxpool_0 = MaxPool2D(pool_size=(sequence_length - conv_size + 1, 1), strides=(1, 1),
                                          padding='valid')(conv_0)
                    flatten = Flatten()(maxpool_0)
                elif layers == 6:  # (2 * (conv + pool) + 2 * fc)
                    conv_0 = Conv2D(num_filters, kernel_size=conv_size, padding='valid',
                                    kernel_initializer='normal', activation='relu')(reshape)
                    maxpool_0 = MaxPool2D(pool_size=default_pool_size, strides=(1, 1), padding='valid')(conv_0)
                    conv_1 = Conv2D(num_filters, kernel_size=(conv_size, embedding_dim - conv_size),
                                    padding='valid', kernel_initializer='normal', activation='relu')(maxpool_0)
                    maxpool_1 = MaxPool2D(pool_size=(sequence_length - 2 * conv_size + 1, 1), strides=(1, 1),
                                          padding='valid')(conv_1)
                    flatten = Flatten()(maxpool_1)
                elif layers == 8:  # (3 * (conv + pool) + 2 * fc)
                    conv_0 = Conv2D(num_filters, kernel_size=conv_size, padding='valid',
                                    kernel_initializer='normal', activation='relu')(reshape)
                    maxpool_0 = MaxPool2D(pool_size=default_pool_size, strides=(1, 1), padding='valid')(conv_0)
                    conv_1 = Conv2D(num_filters, kernel_size=conv_size, padding='valid',
                                    kernel_initializer='normal', activation='relu')(maxpool_0)
                    maxpool_1 = MaxPool2D(pool_size=default_pool_size, strides=(1, 1), padding='valid')(conv_1)
                    conv_2 = Conv2D(num_filters, kernel_size=(conv_size, embedding_dim - 2 * conv_size),
                                    padding='valid', kernel_initializer='normal', activation='relu')(maxpool_1)
                    maxpool_2 = MaxPool2D(pool_size=(sequence_length - 3 * conv_size + 1, 1), strides=(1, 1),
                                          padding='valid')(conv_2)
                    flatten = Flatten()(maxpool_2)
                else:  # layers == 10 (4 * (conv + pool) + 2 * fc)
                    conv_0 = Conv2D(num_filters, kernel_size=conv_size, padding='valid',
                                    kernel_initializer='normal', activation='relu')(reshape)
                    maxpool_0 = MaxPool2D(pool_size=default_pool_size, strides=(1, 1), padding='valid')(conv_0)
                    conv_1 = Conv2D(num_filters, kernel_size=conv_size, padding='valid',
                                    kernel_initializer='normal', activation='relu')(maxpool_0)
                    maxpool_1 = MaxPool2D(pool_size=default_pool_size, strides=(1, 1), padding='valid')(conv_1)
                    conv_2 = Conv2D(num_filters, kernel_size=conv_size, padding='valid',
                                    kernel_initializer='normal', activation='relu')(maxpool_1)
                    maxpool_2 = MaxPool2D(pool_size=default_pool_size, strides=(1, 1), padding='valid')(conv_2)
                    conv_3 = Conv2D(num_filters, kernel_size=(conv_size, embedding_dim - 3 * conv_size),
                                    padding='valid', kernel_initializer='normal', activation='relu')(maxpool_2)
                    maxpool_3 = MaxPool2D(pool_size=(sequence_length - 4 * conv_size + 1, 1), strides=(1, 1),
                                          padding='valid')(conv_3)
                    flatten = Flatten()(maxpool_3)

                # Output layers
                dense1 = Dense(units=num_filters, activation='relu')(flatten)
                dropout = Dropout(dropout_value)(dense1)
                output = Dense(units=len(y[0]), activation='softmax')(dropout)

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
                    './data/models/{}l-{}b-{}c-text_trained.hdf5'.format(layers, batch_size, conv_size),
                    monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
                adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
                model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

                """ Training phase """
                print("Training Model...")
                history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                                    callbacks=[checkpoint, tensorboard],
                                    validation_data=(X_test, y_test))  # starts training


text_train()
