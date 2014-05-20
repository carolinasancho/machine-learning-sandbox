import numpy as np
import scipy
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help='Pydisanto path.', required=True)
parser.add_argument('-s', '--train', help='Train data.', required=True)
parser.add_argument('-t', '--test', help='Test data', required=True)
args = vars(parser.parse_args())

sys.path.append(args['path'])
from pydisanto.supervised.knn import KNNClassifier

train_data = scipy.loadtxt(args['train'], delimiter=',')
labels = train_data[:,-1]
data = train_data[:,:-1]

test_data = scipy.loadtxt(args['test'], delimiter=',')
test_features = test_data[:,:-1]
test_labels = test_data[:,-1]

knn = KNNClassifier(1)
errorCount = 0.0

for i in xrange(len(test_data)):
    result = knn.classify(test_features[i], data, labels)
    if (result != test_labels[i]):
        errorCount += 1.0
    #print test_labes[i]
    #print test_features[i]

print "Accuracy: %f" % (1.0 - (errorCount/float(len(test_data))))
