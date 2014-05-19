import numpy as np
import operator


## TODO: transformar em uma classe
## TODO: criar metodos de classificacao e regressao
## TODO: receber pesos para features?

def createDataset():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def norm(dataset):
    min_vals = dataset.min(0)
    max_vals = dataset.max(0)
    ranges = max_vals - min_vals
    norm_dataset = zeros(shape(dataset))
    m = dataset.shape[0]
    norm_dataset = dataset - tile(min_vals, (m, 1))
    norm_dataset = norm_dataset/tile(ranges, (m, 1))
    return norm_dataset, ranges, min_vals

class KNNClassifier:

    def __init__(self, neighbors=5):
        self.neighbors = neighbors

    def classify(self, input, dataset, labels):
        dataset_size = dataset.shape[0]
        difference_matrix = np.tile(input, (dataset_size, 1)) - dataset
        square_difference_matrix = difference_matrix**2
        square_distances = square_difference_matrix.sum(axis=1)
        distances = square_distances**0.5
        sorted_distance_indices = distances.argsort()
        class_count = {}
        for i in range(self.neighbors):
            vote = labels[sorted_distance_indices[i]]
            class_count[vote] = class_count.get(vote, 0) + 1
        sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sorted_class_count[0][0]
