import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf


t = tf.constant([[[1.1, 1.2, 1.3], [2.1, 2.2, 2.3]],
                 [[3.1, 3.2, 3.3], [4.1, 4.2, 4.3]],
                 [[5.1, 5.2, 5.3], [6.1, 6.2, 6.3]]])
tf.slice(t, [1, 0, 0], [1, 1, 3])  # [[[3, 3, 3]]]
tf.slice(t, [1, 0, 0], [1, 2, 3])  # [[[3, 3, 3],
                                   #   [4, 4, 4]]]
tf.slice(t, [1, 0, 0], [2, 1, 3])  # [[[3, 3, 3]],
                                   #  [[5, 5, 5]]]

vectors = tf.constant([[1.1, 1.2], [2.1, 2.2],
                 [3.1, 3.2], [4.1, 4.2],
                 [5.1, 5.2], [6.1, 6.2]])
centroids = tf.constant([[2.1, 2.2],[3.1, 3.2], [6.1, 6.2]])

expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroids = tf.expand_dims(centroids, 1)
vectors_subtration = tf.subtract(expanded_vectors, expanded_centroids)

# 計算取得距離
euclidean_distances = tf.reduce_sum(tf.square(vectors_subtration), 2)
assignments = tf.to_int32(tf.argmin(euclidean_distances, 0))

# 將資料分區
num_clusters = 3
partitions = tf.dynamic_partition(vectors, assignments, num_clusters)

updata_centroids = tf.concat([tf.expand_dims(tf.reduce_mean(partition, 0), 0)for partition in partitions], 0)

with tf.Session() as sess:
    #print(sess.run(vectors),sess.run(tf.shape(vectors)))
    #print("==================")
    print(sess.run(expanded_vectors),sess.run(tf.shape(expanded_vectors)))
    print("==============================================")

    #print(sess.run(centroids),sess.run(tf.shape(centroids)))
    #print("==================")
    print(sess.run(expanded_centroids),sess.run(tf.shape(expanded_centroids)))
    print("==============================================")
    print(sess.run(vectors_subtration),sess.run(tf.shape(vectors_subtration)))
    print("==============================================")
    print(sess.run(tf.square(vectors_subtration)),sess.run(tf.shape(tf.square(vectors_subtration))))
    print("==============================================")
    print(sess.run(euclidean_distances), sess.run(tf.shape(euclidean_distances)))
    print("==============================================")
    print(sess.run(assignments), sess.run(tf.shape(assignments)))
    print("==============================================")
    print(sess.run(partitions), sess.run(tf.shape(partitions)))
    print("==============================================")
    print(sess.run(updata_centroids), sess.run(tf.shape(updata_centroids)))





