import numpy as np
import tensorflow as tf
from keras import backend as K

a = [[1, 2], [3, 4], [5, 6], [7, 8]]
b = [[0,1],[0,2],[0,3],[0,4],[0,5]]
sess = tf.Session()

with tf.Session() as sess:
    print(sess.run(K.gather(a, b)))