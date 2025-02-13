import configparser

from mnist_benchmark import mnist_bench
from text_benchmark import text_bench

print("Benchmark is starting...")
print("If you get an error, it's because the model you're searching for is not been trained yet.\n")

config = configparser.ConfigParser()
config.read('config.ini')

text_elapsed_times, text_mean_times = text_bench()
mnist_elapsed_times, mnist_mean_times = mnist_bench()

filter_sizes = config.get('CONFIGURATION', 'Filter_Size').split()  # [3, 5, 8]
batch_sizes = config.get('CONFIGURATION', 'Batch_Size').split()  # [1, 4, 16, 64]
num_layers = config.get('CONFIGURATION', 'Tot_Layers').split()  # [4, 6, 8, 10]

print("------------------------------------ TEXT BENCHMARK ------------------------------------\n")
for k in range(0, len(filter_sizes)):
    for j in range(0, len(batch_sizes)):
        for i in range(0, len(num_layers)):
            conv_size = int(filter_sizes[k])
            batch_size = int(batch_sizes[j])
            layers = int(num_layers[i])
            eval_iterations = int(config['CONFIGURATION']['Evaluation_Iterations'])

            print("Text model with {} layers and batch_size = {}, convolution size = {}:".format(layers,
                                                                                                 batch_size,
                                                                                                 conv_size))
            print("Total time for {} iterations: {}s".format(eval_iterations, text_elapsed_times[i + j]))
            print("Mean time per iteration: {}s".format(text_mean_times[i + j]))
            print("")

print("")
print("------------------------------------ MNIST BENCHMARK ------------------------------------\n")
for k in range(0, len(filter_sizes)):
    for j in range(0, len(batch_sizes)):
        for i in range(0, len(num_layers)):
            conv_size = int(filter_sizes[k])
            batch_size = int(batch_sizes[j])
            layers = int(num_layers[i])
            eval_iterations = int(config['CONFIGURATION']['Evaluation_Iterations'])
            print("Mnist model with {} layers and batch_size = {}, convolution size = {}:".format(layers,
                                                                                                  batch_size,
                                                                                                  conv_size))
            print("Total time for {} iterations: {}s".format(eval_iterations, mnist_elapsed_times[i + j]))
            print("Mean time per iteration: {}s".format(mnist_mean_times[i + j]))
            print("")
