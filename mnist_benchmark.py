import configparser
import logging
import pathlib
import time
from datetime import datetime

import keras.models
import tensorflow as tf

from mnist_data_helpers import extract_data
from mnist_data_helpers import extract_labels

tf.get_logger().setLevel(logging.ERROR)


def mnist_bench():
    config = configparser.ConfigParser()
    config.read('config.ini')

    print('Loading data...\n')
    test_data_filename = './data/images/t10k-images-idx3-ubyte.gz'
    test_labels_filename = './data/images/t10k-labels-idx1-ubyte.gz'
    X_test = extract_data(test_data_filename, 10000)
    y_test = extract_labels(test_labels_filename, 10000)

    # Riduce dimension of test data to 2000 samples
    X_test = X_test[:2000]
    y_test = y_test[:2000]
    # X_test.shape -> (2000, 28x28)

    # MNIST parameters
    width = 28
    height = 28
    depth = 1

    X_test = X_test.reshape(X_test.shape[0], height, width, 1)

    categories = ["top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]

    elapsed_times = []  # Total time for all iterations
    mean_times = []  # Mean time for one iteration

    filter_sizes = config.get('CONFIGURATION', 'Filter_Size').split()  # [3, 5, 8]
    batch_sizes = config.get('CONFIGURATION', 'Batch_Size').split()  # [1, 4, 16, 64]
    num_layers = config.get('CONFIGURATION', 'Tot_Layers').split()  # [4, 6, 8, 10]

    """ Models presence check """
    print("\nChecking models presence...")
    models_present = True
    for i in range(0, len(num_layers)):
        for j in range(0, len(batch_sizes)):
            for k in range(0, len(filter_sizes)):
                conv_size = int(filter_sizes[k])
                batch_size = int(batch_sizes[j])
                layers = int(num_layers[i])
                path = pathlib.Path("./data/models/" + str(layers) + "l-" + str(batch_size) +
                                    "b-" + str(conv_size) + "c-mnist_trained.hdf5")
                if not path.exists():
                    print("You need to train a Mnist model with {} layers, batch_size = {} and convolution size = {}"
                          .format(layers,
                                  batch_size,
                                  conv_size))
                    models_present = False
    if not models_present:
        print("\nTrain the models above before performing Mnist benchmark")
        exit()

    bench_log = "benchmarks/mnist-{}.csv".format(datetime.now())
    file = open(bench_log, "w")
    file.write("Layers,BatchSize,ConvSize,TotalTime,MeanTime\n")

    print("\n------------------------------------ MNIST BENCHMARK ------------------------------------\n")
    for i in range(0, len(num_layers)):
        for j in range(0, len(batch_sizes)):
            for k in range(0, len(filter_sizes)):

                conv_size = int(filter_sizes[k])
                batch_size = int(batch_sizes[j])
                layers = int(num_layers[i])

                model = keras.models.load_model("./data/models/" + str(layers) + "l-" + str(batch_size) +
                                                "b-" + str(conv_size) + "c-mnist_trained.hdf5")
                # model = keras.models.load_model("./data/models/mnist_trained.hdf5")  # used for testing
                eval_iterations = int(config['CONFIGURATION']['Evaluation_Iterations'])

                print("")
                print("")
                print("Mnist model with {} layers, batch_size = {} and convolution size = {}".format(layers,
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

                file.write(str(layers) + "," + str(batch_size) + "," +
                           str(conv_size) + "," + str(elapsed_time) + "," +
                           str(mean_time) + "\n")

                """ Print expected category + output """
                # categories = ["top", "trouser", "pullover", "dress", "coat",
                # "sandal", "shirt", "sneaker", "bag", "ankle boot"]
                """
                print("")
                for h in range(0, len(prediction)):
                    actual_label = ""
                    predicted_label = ""
                    for l in range(0, len(y_test[i])):
                        if y_test[h][l] == 1:
                            actual_label = categories[l]
                    max_inference = 0
                    for m in range(0, len(prediction[h])):
                        if prediction[h][m] > max_inference:
                            max_inference = prediction[h][m]
                            predicted_label = categories[m]
                    if actual_label == predicted_label:
                        result = "CORRECT"
                    else:
                        result = "WRONG"
                    print(result)
                    print("Actual: " + actual_label + " | Predicted: " + predicted_label + "\n")
                """
                print("_______________________________________________________________________________________________")

    file.close()
    print("Benchmark data saved in folder /benchmarks.")

    return elapsed_times, mean_times


mnist_bench()
