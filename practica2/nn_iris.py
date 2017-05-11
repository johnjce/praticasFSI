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


data = np.genfromtxt('iris.data', delimiter=",")  # iris.data file loading
np.random.shuffle(data)  # we shuffle the data

x_data = data[:, 0:4].astype('f4')  # the samples are the four first rows of data
y_data = one_hot(data[:, 4].astype(int), 3)  # the labels are in the last row. Then we encode them in one hot code

x_dataTrain = x_data[0:105]
y_dataTrain = y_data[0:105]

x_dataVal = x_data[105:128]
y_dataVal = y_data[105:128]

x_dataTest = x_data[128:]
y_dataTest = y_data[128:]

print ("\nSome samples...")
for i in range(20):
    print (x_data[i], " -> ", y_data[i])
print

x = tf.placeholder("float", [None, 4])  # samples
y_ = tf.placeholder("float", [None, 3])  # labels

W1 = tf.Variable(np.float32(np.random.rand(4, 5)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(5, 3)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(3)) * 0.1)

#h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

print ("----------------------")
print ("   Start training...  ")
print ("----------------------")

batch_size = 20

for epoch in range(100):
    for jj in range(int(len(x_dataTrain) / batch_size)):
        batch_xsTrain = x_dataTrain[jj * batch_size: jj * batch_size + batch_size]
        batch_ysTrain = y_dataTrain[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xsTrain, y_: batch_ysTrain})

    print("....................Training....................")
    print ("Epoch #:", epoch, "Error on train: ", sess.run(loss, feed_dict={x: batch_xsTrain, y_: batch_ysTrain}))

    print("....................Validation....................")
    print("Epoch #:", epoch, "Error on validation: ", sess.run(loss, feed_dict={x: x_dataVal, y_: y_dataVal}))
    resultVal = sess.run(y, feed_dict={x: x_dataVal})
    for b, r in zip(y_dataVal, resultVal):
        print( b, "-->", r)
    print("----------------------------------------------------------------------------------")
print("....................Test....................")
errorsOnTest = 0

resultTest = sess.run(y, feed_dict={x: x_dataTest})
for b, r in zip(y_dataTest, resultTest):
    if np.argmax(b) != np.argmax(r):
        errorsOnTest += 1
    print(b, "-->", r)
print("----------------------------------------------------------------------------------")
print("Errors on test:", errorsOnTest)