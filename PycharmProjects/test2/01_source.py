import tensorflow as tf

a = tf.constant(10, name="a")
b = tf.constant(90, name="b")
y = tf.Variable(a+b*2, name="y")

model = tf.initialize_all_variables()

x1 = tf.constant([[1, 2, 3], [4, 5, 6], [.1, .2, .3]])
x2 = tf.constant([[[1, 2], [4, 5]],[ [.1, .2], [.3, .4]]])
x1_out = tf.reduce_sum(x1, -1)
x2_out = tf.reduce_sum(x2, 0)
x3_out = tf.reduce_sum(x2, 1)
x4_out = tf.reduce_sum(x2, 2)

with tf.Session() as session:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("/Users/mac-yanqi/tensorflowlogs",session.graph)
    session.run(model)
    print(session.run(y))
    print(session.run(x1_out))

    print('out=', session.run(x2_out))
    print('out=',session.run(x3_out))
    print('out=',session.run(x4_out))
