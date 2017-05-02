import numpy as np
import tensorflow as tf

if __name__ == '__main__':
    if __name__ == '__main__':
        if __name__ == "__main__":
            # y = w*x+b 优化线性回归
            x = tf.placeholder(tf.float32)
            y = tf.placeholder(tf.float32)

            w = tf.Variable(3.0, tf.float32)
            b = tf.Variable(-3.0, tf.float32)

            linear_model = w * x + b

            square_sum = tf.square(linear_model - y)
            loss = tf.reduce_sum(square_sum)

            opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
            train = opt.minimize(loss)

            ## init a session
            sess = tf.Session()
            ## initialize all varaibles
            sess.run(tf.global_variables_initializer())

            ## train..
            for i in range(1000):
                sess.run(train, {x:[1,2,3,4],y:[0,-1,-2,-3]})
            print(sess.run([w,b,loss], {x:[1,2,3,4],y:[0,-1,-2,-3]}))

