import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b

adder_node_triple = adder_node * 3
sess = tf.Session()

print(sess.run(adder_node, {a: [1, 2, 3], b: [2, 3, 4]}))
print(sess.run(adder_node_triple, {a: [1, 2, 3], b: [2, 3, 4]}))

