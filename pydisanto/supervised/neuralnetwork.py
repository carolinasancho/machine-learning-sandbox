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

def sigmoid(z):
    return 1.0/(1 + np.exp(-z))

sigmoid_vec = np.vectorize(sigmoid)

def test_nn():
    net = NeuralNetwork([2,3,1])
    print net.biases
    print net.weights
    print len(zip(net.biases, net.weights))


    #print net.biases
    #print net.weights

if __name__ == '__main__':
    test_nn()
