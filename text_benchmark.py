import configparser
import logging
import time

import keras.models
import tensorflow as tf
from sklearn.model_selection import train_test_split

from text_data_helpers import load_data

tf.get_logger().setLevel(logging.ERROR)


def text_bench():
    config = configparser.ConfigParser()
    config.read('config.ini')

    print('Loading data...')
    x, y, vocabulary, vocabulary_inv = load_data()
    categories = ["negative", "positive"]

    # x.shape -> (10662, 56)
    # y.shape -> (10662, 2)
    # len(vocabulary) -> 18765
    # len(vocabulary_inv) -> 18765

    # Split test and train dataset (random_state is the seed used for random number generation)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Riduce dimension of test data to 2000 samples
    X_test = X_test[:2000]
    y_test = y_test[:2000]
    # X_test.shape -> (2000, 56)

    elapsed_times = []  # Total time for all iterations
    mean_times = []  # Mean time for one iteration

    filter_sizes = config.get('CONFIGURATION', 'Filter_Size').split()  # [3, 5, 8]
    batch_sizes = config.get('CONFIGURATION', 'Batch_Size').split()  # [1, 4, 16, 64]
    num_layers = config.get('CONFIGURATION', 'Tot_Layers').split()  # [4, 6, 8, 10]

    print("\n------------------------------------ TEXT BENCHMARK ------------------------------------\n")
    for k in range(0, len(filter_sizes)):
        for j in range(0, len(batch_sizes)):
            for i in range(0, len(num_layers)):

                conv_size = int(filter_sizes[k])
                batch_size = int(batch_sizes[j])
                layers = int(num_layers[i])

                #model = keras.models.load_model("./data/models/" + str(layers) + "l-" + str(batch_size) +
                #                                "b-" + str(conv_size) + "c-text_trained.hdf5")
                model = keras.models.load_model("./data/models/text_trained.hdf5")  # used for testing

                eval_iterations = int(config['CONFIGURATION']['Evaluation_Iterations'])

                print("")
                print("")
                print("Text model with {} layers and batch_size = {}, convolution size = {}:".format(layers,
                                                                                                     batch_size,
                                                                                                     conv_size))
                print("Evaluation starting...")
                print("Iterations of evaluation: " + str(eval_iterations))
                print("Inputs to predict for each iteration: " + str(len(X_test)))
                start_time = time.time()
                for e in range(0, eval_iterations):
                    prediction = model.predict(X_test)
                elapsed_time = time.time() - start_time
                mean_time = elapsed_time / eval_iterations
                print("Total elapsed time: " + str(elapsed_time) + "s")
                print("Prediction time: " + str(mean_time) + "s")
                print("")

                elapsed_times.append(elapsed_time)
                mean_times.append(mean_time)

                """ Print sentences + output """
                """
                print("")
                for m in range(0, len(prediction)):
                    sentence = ""
                    for n in range(0, len(X_test[m])):
                        if vocabulary_inv[X_test[m][n]] != "<PAD/>":
                            if n != 0:
                                sentence += " "
                            sentence += vocabulary_inv[X_test[m][n]]
                    act_index = list(map(lambda l: l > 0.5, y_test[m])).index(True)
                    pred_index = list(map(lambda l: l > 0.5, prediction[m])).index(True)
                    if pred_index == act_index:
                        result = "CORRECT"
                    else:
                        result = "WRONG"
                    print(result)
                    print("Actual: " + categories[act_index] + " | Predicted: " + categories[pred_index])
                    print("Review: " + sentence + "\n")
                """
                print("_______________________________________________________________________________________________")

    return elapsed_times, mean_times


text_bench()
