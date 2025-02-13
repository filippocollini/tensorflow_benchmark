import gzip
import os

import numpy
import tensorflow as tf
from scipy import ndimage
from six.moves import urllib

source = 'http://yann.lecun.com/exdb/mnist/'
data_directory = "data"

# MNIST images parameters
image_size = 28
num_channels = 1
pixel_depth = 255
num_labels = 10
validation_size = 10000


# Download MNIST data
def maybe_download(filename):
    """Download the data from Yann's website, unless it's already here."""
    if not tf.gfile.Exists(data_directory):
        tf.gfile.MakeDirs(data_directory)
    filepath = os.path.join(data_directory, filename)
    if not tf.gfile.Exists(filepath):
        filepath, _ = urllib.request.urlretrieve(source + filename, filepath)
        with tf.gfile.GFile(filepath) as f:
            size = f.size()
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath


# Extract the images
def extract_data(filename, num_images):
    """
    Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [0, 1].
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(image_size * image_size * num_images * num_channels)
        data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
        data = data / pixel_depth
        data = data.reshape(num_images, image_size, image_size, num_channels)
        data = numpy.reshape(data, [num_images, -1])
    return data


# Extract the labels
def extract_labels(filename, num_images):
    """ Extract the labels into a vector of int64 label IDs. """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
        num_labels_data = len(labels)
        one_hot_encoding = numpy.zeros((num_labels_data, num_labels))
        one_hot_encoding[numpy.arange(num_labels_data), labels] = 1
        one_hot_encoding = numpy.reshape(one_hot_encoding, [-1, num_labels])
    return one_hot_encoding


# Augment training data
def expend_training_data(images, labels):
    expanded_images = []
    expanded_labels = []

    j = 0  # counter
    for x, y in zip(images, labels):
        j = j + 1
        if j % 100 == 0:
            print('expanding data : %03d / %03d' % (j, numpy.size(images, 0)))

        # register original data
        expanded_images.append(x)
        expanded_labels.append(y)

        # get a value for the background
        # zero is the expected value, but median() is used to estimate background's value
        bg_value = numpy.median(x)  # this is regarded as background's value
        image = numpy.reshape(x, (-1, 28))

        for i in range(4):
            # rotate the image with random degree
            angle = numpy.random.randint(-15, 15, 1)
            new_img = ndimage.rotate(image, angle, reshape=False, cval=bg_value)

            # shift the image with random distance
            shift = numpy.random.randint(-2, 2, 2)
            new_img_ = ndimage.shift(new_img, shift, cval=bg_value)

            # register new training data
            expanded_images.append(numpy.reshape(new_img_, 784))
            expanded_labels.append(y)

    # images and labels are concatenated for random-shuffle at each epoch
    # notice that pair of image and label should not be broken
    expanded_train_total_data = numpy.concatenate((expanded_images, expanded_labels), axis=1)
    numpy.random.shuffle(expanded_train_total_data)

    return expanded_train_total_data


# Prepare MNISt data
def prepare_MNIST_data(use_data_augmentation=True):
    # Get the data.
    train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
    train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
    test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

    # Extract it into numpy arrays.
    train_data = extract_data(train_data_filename, 60000)
    train_labels = extract_labels(train_labels_filename, 60000)
    test_data = extract_data(test_data_filename, 10000)
    test_labels = extract_labels(test_labels_filename, 10000)

    # Generate a validation set.
    validation_data = train_data[:validation_size, :]
    validation_labels = train_labels[:validation_size, :]
    train_data = train_data[validation_size:, :]
    train_labels = train_labels[validation_size:, :]

    # Concatenate train_data & train_labels for random shuffle
    if use_data_augmentation:
        train_total_data = expend_training_data(train_data, train_labels)
    else:
        train_total_data = numpy.concatenate((train_data, train_labels), axis=1)

    train_size = train_total_data.shape[0]

    return train_total_data, train_size, validation_data, validation_labels, test_data, test_labels
