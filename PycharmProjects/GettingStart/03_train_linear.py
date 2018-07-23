
import tensorflow as tf

W = tf.Variable([-.3], dtype=tf.float32)
b = tf.Variable([.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

linear_model = W*x + b

squared_deltas = tf.square(linear_model - y)

loss = tf.reduce_sum(squared_deltas)

optimizer = tf.train.GradientDescentOptimizer(0.02)
train = optimizer.minimize(loss)

x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})
    #print(i, sess.run(W), sess.run(b), sess.run(loss, {x: x_train, y: y_train}))
    print("W: %s b: %s loss: %s" % (sess.run(W), sess.run(b), sess.run(loss, {x: x_train, y: y_train})))