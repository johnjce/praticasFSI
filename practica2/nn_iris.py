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

data = np.genfromtxt('iris.data', delimiter=",")    # lee el fichero y lo guarda en data como array
np.random.shuffle(data)                             # desordena los datos para leer aleatoreamente

x_data = data[:, 0:4].astype('f4')                  # cojo las 4 primeras lineas
y_data = one_hot(data[:, 4].astype(int), 3)         # codifico con one_hot


# separo la muestras del archivo para aprendizaje, validacion y test
# tenemos 150 muestras en total

xDataTrain = x_data[0:105]   # 150 * 0.7 = 105                      =>70%
yDataTrain = y_data[0:105]

xDataValue = x_data[105:127]  # 150 * 0.3 = 45 / 2 = 22.5 -> 22      =>15%
yDataValue = y_data[105:127]

xDataTest = x_data[127:150] # 150 * 0.3 = 45 / 2 = 22.5 -> 23      =>15%
yDataTest = y_data[127:150]

print("\nSome samples...")
for i in range(20):
    print(x_data[i], " -> ", y_data[i])
print

x = tf.placeholder("float", [None, 4])      # entradas
y_ = tf.placeholder("float", [None, 3])     # salidas esperadas

W1 = tf.Variable(np.float32(np.random.rand(4, 5)) * 0.1)    # omega de la neurona
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)       # beta para afinar la red en las entradas

W2 = tf.Variable(np.float32(np.random.rand(5, 3)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(3)) * 0.1)

h = tf.matmul(x, W1) + b1                   # x * w + b
y = tf.nn.softmax(tf.matmul(h, W2) + b2)    # sumatorio Xn * Wn + Bn

loss = tf.reduce_sum(tf.square(y_ - y))     # margen de error

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # hago pasar el entrenamiento por todos los valores
                                                                # minimizando el gradiente

init = tf.global_variables_initializer()                        # inicializo el modelo descrito

sess = tf.Session()                                             # inicio sesion
sess.run(init)                                                  # ejecuto la sesion


batchSize = 20

for epoch in range(100):
    for conter in range(int(len(xDataTrain) / batchSize)):
        batchXsTrain = xDataTrain[conter * batchSize: conter * batchSize + batchSize]
        batchYsTrain = yDataTrain[conter * batchSize: conter * batchSize + batchSize]
        sess.run(train, feed_dict={x: batchXsTrain, y_: batchYsTrain})

    print("Entrenamiento")
    print("Epóca: ", epoch,
          "Errores durante entrenamiento: ", sess.run(loss, feed_dict={x: batchXsTrain, y_: batchYsTrain}))

    print("Validación")
    print("Epóca: ", epoch,
          "Error de validación: ", sess.run(loss, feed_dict={x: xDataValue, y_: yDataValue}))

    result = sess.run(y, feed_dict={x: xDataValue})
    for b, r in zip(yDataValue, result):
        print(b, "<=>", r)                   # muestro los resultados del entrenamiento y la validacion para comparar

print("Test")
errorsInTest = 0

resultOfTest = sess.run(y, feed_dict={x: xDataTest})
for b, r in zip(yDataTest, resultOfTest):
    if np.argmax(b) != np.argmax(r):
        errorsInTest += 1
    print(b, "<=>", r)
print("Errores de test:", errorsInTest)
