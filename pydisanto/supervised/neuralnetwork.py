## THIS CODE IS STRONGLY INSPIRED by http://neuralnetworksanddeeplearning.com/

import numpy as np

class NeuralNetwork():

    def __init__(self, sizes):
        # sizes is a list with the numbers of neurons in each layer
        self.sizes = sizes
        # each element of the sizes is a layer.
        self.num_layers = len(sizes)
        # for each layer one bias for each neuron
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # for each layer, y neurons receives x inputs.
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid_vec(np.dot(w,a)+b)
        return a

    def sgd(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if (test_data):
            n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {}: {} / {}".format(j, self.evaluate(test_data), n_test)
            else:
                print "Epoch %s complete" % j

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        ## TODO 

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

sigmoid_vec = np.vectorize(sigmoid)

def test_nn():
    net = NeuralNetwork([2,3,1])
    i =  np.array([[2,2]])
    print net.feedforward(i.T)

if __name__ == '__main__':
    test_nn()
