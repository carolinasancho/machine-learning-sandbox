import numpy

class Perceptron(object):

    def __init__(self, learning_rate=0.25, niterations=10):
        self.learning_rate = learning_rate
        self.number_of_iterations = niterations

    def _initWeights(self, X, Y):
        if (numpy.ndim(X) > 1):
            self.number_of_features = numpy.shape(X)[1]
        else:
            self.number_of_features = 1
        if (numpy.ndim(Y) > 1):
            self.number_of_classes = numpy.shape(Y)[1]
        else:
            self.number_of_classes = 1
        self.train_size = numpy.shape(X)[0]
        self.weights = numpy.random.rand(self.number_of_features+1,self.number_of_classes)*0.1-0.05

    def learn(self,X,Y):
        self._initWeights(X,Y)
        # add the inputs to the bias node
        X = numpy.concatenate((X, -numpy.ones((self.train_size,1))), axis=1)
        change = range(self.train_size)
        for i in xrange(self.number_of_iterations):
            self.outputs = self._forward(X)
            self.weights += self.learning_rate*numpy.dot(X.T,Y-self.outputs)
            numpy.random.shuffle(change)
            X = X[change,:]
            Y = Y[change,:]

    def _forward(self, X):
        outputs = numpy.dot(X, self.weights)
        return numpy.where(outputs > 0, 1, 0)

    def predict(self, X):
        X = numpy.concatenate((X, -numpy.ones((self.train_size,1))), axis=1)
        return self._forward(X)

def main():
    # AND logic function example 
    a = numpy.array([[0,0,0],[0,1,0],[1,0,0],[1,1,1]])
    p  = Perceptron()
    p.learn(a[:,0:2],a[:,2:])
    print p.predict(a[:,0:2])

if __name__ == '__main__':
    main()
