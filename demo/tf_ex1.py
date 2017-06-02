import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0, tf.float32)
#node3 = tf.constant(5.0, tf.float32)

# print(node1, node2, node3)
# Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32) Tensor("Const_2:0", shape=(), dtype=float32)
sess = tf.Session()
#print(sess.run([node1, node2]))
# [3.0, 4.0]

node3 = tf.add(node1, node2)
#print("node3: ", node3)
#print("sess.run(node3): ", sess.run(node3))
#node3:  Tensor("Add:0", shape=(), dtype=float32)
#sess.run(node3):  7.0


## placeholder

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b # + provides a shortcut for tf.add(a,b)
#print(sess.run(adder_node, {a:3, b:4.5}))
#print(sess.run(adder_node, {a:[1,3], b:[2,4]}))
#7.5
#[ 3.  7.]

W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
init = tf.global_variables_initializer()
sess.run(init)
#print(sess.run(linear_model, {x:[1,2,3,4]}))
#[ 0.          0.30000001  0.60000002  0.90000004]

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
#print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
# 23.66

fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
#print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
# 0.0

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init)
for i in range(1000):
    sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

print(sess.run([W, b]))
#[array([-0.9999969], dtype=float32), array([ 0.99999082], dtype=float32)]

