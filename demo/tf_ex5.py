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
y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.add(tf.matmul(x, W),b)) # [None, 10]

## loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis=1))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

## model train
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

batch_size = 100
for epoch in range(10):
    i = 0
    while True:
        end = i + batch_size
        if end >= mnist[0].images.shape[0]:
            end = mnist[0].images.shape[0]
        xs = mnist[0].images[i:end]
        ys = mnist[0].labels[i:end]

        print("traing: epoch: %d xs:%d~%d" % (epoch, i, end))
        sess.run(train_step, feed_dict={x:xs, y_:ys})
        i = end
        if i >= mnist[0].images.shape[0]:
            break
## model test
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1) , tf.argmax(y_, 1)), tf.float32))
print(sess.run(accuracy, feed_dict={x:mnist[1].images, y_:mnist[1].labels}))
