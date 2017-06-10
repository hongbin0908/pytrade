# https://www.tensorflow.org/get_started/mnist/beginners

## prepare data
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


print("train date shape: features: %s, labels: %s" % (str(mnist[0].images.shape), str(mnist[0].labels.shape)))
print("test  date shape: features: %s, labels: %s" % (str(mnist[1].images.shape), str(mnist[1].labels.shape)))
print("valid date shape: features: %s, labels: %s" % (str(mnist[2].images.shape), str(mnist[2].labels.shape)))
#train date shape: features: (55000, 784), labels: (55000, 10)
#test  date shape: features: (5000, 784), labels: (5000, 10)
#valid date shape: features: (10000, 784), labels: (10000, 10)

## model select
x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros[])


## model train

## model test
