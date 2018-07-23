import input_data
import numpy as np
import tensorflow as tf

mnist_images = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_pixels, train_list_values = mnist_images.train.next_batch(100)
test_pixels, test_list_values = mnist_images.train.next_batch(1)

train_pixel_tensor = tf.placeholder("float", [None, 784])
test_pixel_tensor = tf.placeholder("float", [784])

distance = tf.reduce_sum(tf.abs(tf.add(train_pixel_tensor
                                       , tf.negative(test_pixel_tensor)))
                         ,reduction_indices=1)
pred = tf.argmin(distance, 0)
# 將test_data與train＿data進行距離演算，回傳與train_data最短（相近）的index值
accuracy = 0.0

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for i in range(len(test_list_values)):
        nn_index = sess.run(pred, feed_dict={train_pixel_tensor: train_pixels
                                            , test_pixel_tensor: test_pixels[i, :]})
        print("Test N ", i
              , "Predicted Class: ", np.argmax(train_list_values[nn_index])
              , "True Class: ", np.argmax(test_list_values[i]))

        if np.argmax(train_list_values[nn_index]) == np.argmax(test_list_values[i]):
            accuracy += 1./len(test_pixels)
    print("Result = ",accuracy)

