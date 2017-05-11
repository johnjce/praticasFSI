import gzip
import _pickle

import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = _pickle.load(f, encoding='latin1')
f.close()
print(train_set)
xDataTrain, yttr = train_set
yDataTrain = one_hot(yttr, 10)

xDataValid, ytv = valid_set
yDataValid = one_hot(ytv, 10)

xDataTest, ytte = test_set
yDataTest = one_hot(ytte, 10)

# TODO: the neural net!!
x = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

batchSize = 20
actualLoss = 9999
previousLoss = 9999
epoch = 0
while actualLoss <= previousLoss:
    epoch += 1
    for conter in range(int(len(xDataTrain) / batchSize)):
        batch_xsTrain = xDataTrain[conter * batchSize: conter * batchSize + batchSize]
        batch_ysTrain = yDataTrain[conter * batchSize: conter * batchSize + batchSize]
        sess.run(train, feed_dict={x: batch_xsTrain, y_: batch_ysTrain})
    previousLoss = actualLoss
    actualLoss = sess.run(loss, feed_dict={x: xDataValid, y_: yDataValid})
    print("Epóca: ", epoch, " Error actual: ", actualLoss, "Error Anterior: ", previousLoss)

misses = 0
result = sess.run(y, feed_dict={x: xDataTest})
for b, r in zip(yDataTest, result):
    if np.argmax(b) != np.argmax(r):
        misses += 1
print("Porcentaje de error: ", misses / len(xDataTest), "Total: ", misses)
