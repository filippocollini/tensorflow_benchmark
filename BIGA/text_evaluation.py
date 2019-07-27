import keras.models
import time
import configparser
from sklearn.model_selection import train_test_split
from BIGA.text_data_helpers import load_data
from keras.utils import plot_model

config = configparser.ConfigParser()
config.read('config.ini')

print('Loading data')
x, y, vocabulary, vocabulary_inv = load_data()
categories = ["negative", "positive"]

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

model = keras.models.load_model("./data/models/text_trained.hdf5")
plot_model(model, to_file='./data/models/{layers:d}l-{batch_size:d}b-text_trained.png')

eval_iterations = int(config['CONFIGURATION']['Evaluation_Iterations'])

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

""" Print sentences + output """
"""
print("")
for i in range(0, len(prediction)):
    sentence = ""
    for j in range(0, len(X_test[i])):
        if vocabulary_inv[X_test[i][j]] != "<PAD/>":
            if j != 0:
                sentence += " "
            sentence += vocabulary_inv[X_test[i][j]]
    print(sentence)
    index = list(map(lambda l: l > 0.5, prediction[i])).index(True)
    print(categories[index] + "\n")  # prints the categories
"""
