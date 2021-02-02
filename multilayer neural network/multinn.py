# Nie, Golay
# 2019-03-22
# Assignment-03-01

# %tensorflow_version 2.x
# %tensorflow_version 2.x
import tensorflow as tf
import numpy as np


class MultiNN(object):
    def __init__(self, input_dimension):
        self.input_dimension = input_dimension
        self.weights = []
        self.biases = []
        self.activations = []
        self.loss = None

    def add_layer(self, num_nodes, transfer_function="Linear"):
        if not self.weights:
            npWeights = np.array([[np.random.normal() for column in range(num_nodes)] for row in range(self.input_dimension)])
        else:
            npWeights = np.array([[np.random.normal() for column in range(num_nodes)] for row in range(self.weights[-1].shape[1])])
        tfWeights = tf.Variable(npWeights, trainable=True)
        npBias = np.array([[np.random.normal() for column in range(num_nodes)]])
        tfBias = tf.Variable(npBias, trainable=True)
        self.weights.append(tfWeights)
        self.biases.append(tfBias)
        self.activations.append(transfer_function)

    def get_weights_without_biases(self, layer_number):
        return self.weights[layer_number]

    def get_biases(self, layer_number):
        return self.biases[layer_number]

    def set_weights_without_biases(self, weights, layer_number):
        self.weights[layer_number] = weights

    def set_biases(self, biases, layer_number):
        self.biases[layer_number] = biases
		
    def set_loss_function(self, loss_fn):
        self.loss = loss_fn

    def Sigmoid(self, x):
        return tf.nn.sigmoid(x)

    def Linear(self, x):
        return x

    def Relu(self, x):
        out = tf.nn.relu(x)
        return out

    def calculate_loss(self, y, y_hat):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_hat))

    def predict(self, X):
        yhat = tf.Variable(X)
        for layer in range(len(self.weights)):
            weightedX = tf.matmul(yhat, self.get_weights_without_biases(layer), name="WeightedX")
            biasedX = tf.add(weightedX, self.get_biases(layer), "BiasedX")
            yhat = self.activations[layer](biasedX)
        return yhat

    def train(self, X_train, y_train, batch_size, num_epochs, alpha=0.8):
        for epoch in range(num_epochs):
            for sample in range(0, X_train.shape[0], batch_size):
                end_row = sample + batch_size
                if end_row > X_train.shape[0]:
                    end_row = X_train.shape[0]

                sampleX = tf.Variable(X_train[sample : end_row,:])
                sampleY = tf.Variable(y_train[sample : end_row])

                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(self.weights)
                    tape.watch(self.biases)
                    tfResults = self.predict(sampleX)
                    loss = self.calculate_loss(sampleY, tfResults)

                for layer in range(len(self.weights)):
                    dl_dw = tape.gradient(loss, self.get_weights_without_biases(layer))
                    dl_db = tape.gradient(loss, self.get_biases(layer))
                    scaled_dl_dw = tf.scalar_mul(alpha, dl_dw)
                    scaled_dl_db = tf.scalar_mul(alpha, dl_db)
                    new_weights = tf.subtract(self.get_weights_without_biases(layer), scaled_dl_dw)
                    new_bias = tf.subtract(self.get_biases(layer), scaled_dl_db)
                    self.set_weights_without_biases(new_weights, layer)
                    self.set_biases(new_bias, layer)

    def calculate_percent_error(self, X, y):
        result = self.predict(X).numpy()
        one_hot_expected = self.toOneHot(y).transpose()
        max_index = result.argmax(axis=1)
        one_hot_result = (max_index[:, None] == np.arange(result.shape[1])).astype(float)
        errors = 0
        for sample in range(result.shape[0]):
            e = one_hot_expected[sample]
            x = one_hot_result[sample]
            if not np.allclose(e, x):
                errors += 1
        percent_error = errors / result.shape[0]
        return percent_error


    def calculate_confusion_matrix(self, X, y):
        confusion = np.zeros((self.weights[-1].shape[1], self.weights[-1].shape[1]))
        result = self.predict(X).numpy()
        one_hot_expected = self.toOneHot(y).transpose()
        max_index = result.argmax(axis=1)
        one_hot_result = (max_index[:, None] == np.arange(result.shape[1])).astype(float)
        for sample in range(one_hot_result.shape[0]):
            indices = np.where(one_hot_result[sample] == 1)

            if indices[0].size != 0:
                predClass = indices[0][0]

                if not np.array_equal(one_hot_result[sample], one_hot_expected[sample]):
                    confusion[y[0], predClass] += 1
                else:
                    confusion[predClass, predClass] += 1

        return confusion


    def toOneHot(self, Y):
        oneHot = np.zeros((self.weights[-1].shape[1], Y.shape[0]))

        oneHot[Y, np.arange(Y.shape[0])] = 1

        return oneHot

if __name__ == "__main__":
    from tensorflow.keras.datasets import mnist
    np.random.seed(seed=1)
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 784).astype(np.float64) / 255.0 - 0.5
    y_train = y_train.flatten().astype(np.int32)
    input_dimension = X_train.shape[1]
    indices = list(range(X_train.shape[0]))
    number_of_samples_to_use = 500
    X_train = X_train[indices[:number_of_samples_to_use]]
    y_train = y_train[indices[:number_of_samples_to_use]]
    multi_nn = MultiNN(input_dimension)
    number_of_classes = 10
    activations_list = [multi_nn.sigmoid, multi_nn.sigmoid, multi_nn.linear]
    number_of_neurons_list = [50, 20, number_of_classes]
    for layer_number in range(len(activations_list)):
        multi_nn.add_layer(number_of_neurons_list[layer_number], transfer_function=activations_list[layer_number])
    for layer_number in range(len(multi_nn.weights)):
        W = multi_nn.get_weights_without_biases(layer_number)
        W = tf.Variable((np.random.randn(*W.shape)) * 0.1, trainable=True)
        multi_nn.set_weights_without_biases(W, layer_number)
        b = multi_nn.get_biases(layer_number=layer_number)
        b = tf.Variable(np.zeros(b.shape) * 0, trainable=True)
        multi_nn.set_biases(b, layer_number)
    multi_nn.set_loss_function(multi_nn.calculate_loss)
    percent_error = []
    for k in range(10):
        multi_nn.train(X_train, y_train, batch_size=100, num_epochs=20, alpha=0.8)
        percent_error.append(multi_nn.calculate_percent_error(X_train, y_train))
    confusion_matrix = multi_nn.calculate_confusion_matrix(X_train, y_train)
    print("Percent error: ", np.array2string(np.array(percent_error), separator=","))
    print("************* Confusion Matrix ***************\n", np.array2string(confusion_matrix, separator=","))