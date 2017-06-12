from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import urllib.request
import pandas as pd

import numpy as np
import tensorflow as tf

# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

def main():
    # If the training and test sets aren't stored locally, download them.
    if not os.path.exists(IRIS_TRAINING):
        raw = urllib.request.urlopen(IRIS_TRAINING_URL).read().decode('utf8')
        with open(IRIS_TRAINING, "w") as f:
            f.write(raw)

    if not os.path.exists(IRIS_TEST):
        raw = urllib.request.urlopen(IRIS_TEST_URL).read().decode('utf8')
        with open(IRIS_TEST, "w") as f:
            f.write(raw)

    # Load datasets.
    training_set = pd.read_csv(IRIS_TRAINING)
    training_set.columns = ["sl", 'sw', 'pl', 'pw', 'label']
    test_set = pd.read_csv(IRIS_TEST)
    test_set.columns = ["sl", 'sw', 'pl', 'pw', 'label']
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=[10, 20, 10],
                                                n_classes=3,
                                                model_dir="/tmp/iris_model")
    # Define the training inputs
    def get_train_inputs():
      x = tf.constant(training_set[["sl", 'sw', 'pl', 'pw']].values)
      y = tf.constant(training_set['label'].values)

      return x, y

    # Fit model.
    classifier.fit(input_fn=get_train_inputs, steps=2000)

    # Define the test inputs
    def get_test_inputs():
      x = tf.constant(test_set[["sl", 'sw', 'pl', 'pw']].values)
      y = tf.constant(test_set['label'].values)

      return x, y

    # Evaluate accuracy.
    accuracy_score = classifier.evaluate(input_fn=get_test_inputs, steps=1)["accuracy"]

    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

    # Classify two new flower samples.
    def new_samples():
      return np.array(
        [[6.4, 3.2, 4.5, 1.5],
         [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)

    predictions = list(classifier.predict(input_fn=new_samples))

    print(
        "New Samples, Class Predictions:    {}\n"
        .format(predictions))

if __name__ == "__main__":
    main()