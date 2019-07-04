# tensorflow_benchmark
### Name:     Tensorflow Neural Network Complexity Scaling
### Code:    P17
### Type:        Programming(Python)
### Max Points:    12 (6 + 6)

## Description:
Tensorflow being a high-level python framework mainly for Neural Network applications
may behave quite differently based on the complexity of application requirements. The
main concern is upon reaching some certain thresholds, it may not perform well or its
performance scaling may not be as desirable as before. Therefore, the objective of this
project is the explore and discover the ranges of parameters related to computation
complexity and make an analysis on the scaling of the execution time of the inference
phase (not the training phase).
The generation of benchmarks must be automated as much as possible and it must be
configurable with a config-file. The suggested degrees of freedoms are ”number of layers”, ”size of convolutions”, ”size of input”, ”batching of input”, etc. The expected deliverables of the project are:
Configuration files for benchmark generation.
Python scripts that run the Tensorflow benchmarks.
Analysis results with graphs to justify the claims.

### References (Starting Points):
https://www.tensorflow.org/
https://www.tensorflow.org/tutorials/layers#intro_to_convolutional_neural_networks
https://www.tensorflow.org/tutorials/deep_cnn#highlights_of_the_tutorial

