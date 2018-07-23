import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

def display_partiton(x_values, y_values, assignment_values):
    labels = []
    colors = ['red', 'blue', 'green', 'yellow']
    for i in range(len(assignment_values)):
        labels.append(colors[assignment_values[i]])
    color = labels

    df = pd.DataFrame(dict(x=x_values, y=y_values, color=labels))
    fig, ax = plt.subplots()
    ax.scatter(df['x'], df['y'], c=df['color'])
    plt.show()


num_vectors = 2000
num_clusters = 4
n_samples_per_cluster = 500
num_steps = 100

x_values = []
y_values = []
vector_values = []

for i in range(num_vectors):
    if np.random.random() > 0.5:
        x_values.append(np.random.normal(0.4, 0.7))
        y_values.append(np.random.normal(0.2, 0.8))
    else:
        x_values.append(np.random.normal(0.6, 0.4))
        y_values.append(np.random.normal(0.8, 0.5))
# 將Ｘ陣列與Ｙ陣列合併為一項量陣列-> [[x0,y0],[x1,y1]..]
vector_values = list(zip(x_values, y_values))
vectors = tf.constant(vector_values)
# 取得陣列維度大小
n_samples = tf.shape(vector_values)[0]
# 製作一n_samples數量的編號陣列，範圍為0~n_samples-1
sample_range = tf.range(0, n_samples)
# 將編號陣列進行洗牌
random_indices = tf.random_shuffle(sample_range)

begin = [0, ]
size = [num_clusters, ]
size[0] = num_clusters

# 從洗牌後的編號陣列中取出num_clusters個值，型態為[a1,a2,a3,...]
centroid_indices = tf.slice(random_indices, begin, size)
# 再利用上述值從原陣列中取得num_clusters個中心點，型態為 [[xa,ya],[xb,yb],...]
centroids = tf.Variable(tf.gather(vector_values, centroid_indices))

# 增加維度
expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroids = tf.expand_dims(centroids, 1)
# 取得所有資料與取樣的的差，型態為[[[xa0,ya0],[xa1,ya1]..],[[xb0,yb0],[xb1,yb1]..],...] shape=[num_clusters,原資料數,2]
vectors_subtration = tf.subtract(expanded_vectors, expanded_centroids)

# 計算取得距離
euclidean_distances = tf.reduce_sum(tf.square(vectors_subtration), 2)
# 取得最短距離，以獲得每個資料點與參考點最近的參考點資訊
assignments = tf.to_int32(tf.argmin(euclidean_distances, 0))

# 將資料進行分區
partitions = tf.dynamic_partition(vectors, assignments, num_clusters)
# print([tf.expand_dims(tf.reduce_mean(partition, 0), 0)for partition in partitions])
# updata_centroids = tf.concat([tf.expand_dims(tf.reduce_mean(partition, 0), 0)for partition in partitions], 0)

init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)
for step in range(num_steps):
    centroid_values, assignment_values = sess.run([ centroids, assignments])

display_partiton(x_values, y_values, assignment_values)
plt.plot(x_values, y_values, 'o', label='Input Data')
plt.plot(centroid_values[:, 0], centroid_values[:, 1], 'ro', label='central Data')

print(centroid_values[:, 1])
# label setting
plt.legend()
plt.show()

