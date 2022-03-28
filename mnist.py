import numpy as np
import tensorflow as tf
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

np.savez('./mnist.npz',x_train, y_train, x_test, y_test)