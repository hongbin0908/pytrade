# https://www.tensorflow.org/get_started/mnist/beginners

## prepare data
import numpy as np
from main.classifier.tf_dnn import TfDnn

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)


print("train date shape: features: %s, labels: %s" % (str(mnist[0].images.shape), str(mnist[0].labels.shape)))
print("test  date shape: features: %s, labels: %s" % (str(mnist[1].images.shape), str(mnist[1].labels.shape)))
print("valid date shape: features: %s, labels: %s" % (str(mnist[2].images.shape), str(mnist[2].labels.shape)))

#train date shape: features: (55000, 784), labels: (55000, 10)
#test  date shape: features: (5000, 784), labels: (5000, 10)
#valid date shape: features: (10000, 784), labels: (10000, 10)

## model test

tf_dnn = TfDnn()
tf_dnn.fit(mnist[0].images, mnist[0].labels, None, None)
pred = tf_dnn.predict_proba(mnist[1].images).argmax(axis=1)
print(type(pred))
matched = np.where(np.equal(pred, mnist[1].labels))

print(pred[0:20])
print(mnist[1].labels[0:20])
print(np.size(matched, axis=1)/np.size(pred))
