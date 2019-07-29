import configparser
import time

import keras.models

from mnist_data_helpers import extract_data


def mnist_bench():
    config = configparser.ConfigParser()
    config.read('config.ini')

    print('Loading data...\n')
    test_data_filename = './data/images/t10k-images-idx3-ubyte.gz'
    X_test = extract_data(test_data_filename, 10000)

    # Riduce dimension of test data to 2000 samples
    X_test = X_test[:2000, :, :]
    # X_test.shape -> (2000, 28x28)

    # MNIST parameters
    width = 28
    height = 28
    depth = 1

    X_test = X_test.reshape(X_test.shape[0], height, width, 1)

    categories = ["top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]

    elapsed_times = []  # Total time for all iterations
    mean_times = []  # Mean time for one iteration

    batch_sizes = config.get('CONFIGURATION', 'Batch_Size').split()  # [1, 4, 16, 64]
    num_layers = config.get('CONFIGURATION', 'Tot_Layers').split()  # [4, 6, 8, 10]

    print("------------------------------------ MNIST BENCHMARK ------------------------------------\n")
    for j in range(0, len(batch_sizes)):
        for i in range(0, len(num_layers)):

            batch_size = int(batch_sizes[j])
            layers = int(num_layers[i])

            model = keras.models.load_model("./data/models/{layers:d}l-{batch_size:d}b-mnist_trained.hdf5")

            eval_iterations = int(config['CONFIGURATION']['Evaluation_Iterations'])

            print("Model with {} layers and batch_size = {}:".format(layers, batch_size))
            print("Evaluation starting...")
            print("Iterations of evaluation: " + str(eval_iterations))
            print("Inputs to predict for each iteration: " + str(len(X_test)))
            print("")
            start_time = time.time()
            for e in range(0, eval_iterations):
                prediction = model.predict(X_test)
            elapsed_time = time.time() - start_time
            mean_time = elapsed_time / eval_iterations
            print("Total elapsed time: " + str(elapsed_time) + "s")
            print("Prediction time: " + str(mean_time) + "s")

            elapsed_times.append(elapsed_time)
            mean_times.append(mean_time)

            """ Print sentences + output """
            # TODO

    return elapsed_times, mean_times


mnist_bench()
