import numpy as np
import operator


## TODO: features weight?
## TODO: add other distances measures (manhattan)
## TODO: add a super class to generalize the neighbor search

def createDataset():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

class KNN:

    def nearest_neighbors(self, input, dataset):
        dataset_size = dataset.shape[0]
        difference_matrix = np.tile(input, (dataset_size, 1)) - dataset
        square_difference_matrix = difference_matrix**2
        square_distances = square_difference_matrix.sum(axis=1)
        distances = square_distances**0.5
        sorted_distance_indices = distances.argsort()
        return sorted_distance_indices


class KNNClassifier(KNN):

    def __init__(self, neighbors=5):
        self.neighbors = neighbors

    def classify(self, input, dataset, labels):
        sorted_distance_indices = self.nearest_neighbors(input, dataset)
        class_count = {}
        for i in range(self.neighbors):
            vote = labels[sorted_distance_indices[i]]
            class_count[vote] = class_count.get(vote, 0) + 1
        sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sorted_class_count[0][0]

class KNNRegressor:

    def __init__(self, neighbors=5):
        self.neighbors = neighbors



def test_knnclassifier():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    knn = KNNClassifier(3)
    print knn.classify([0,0], group, labels)

if __name__ == '__main__':
    test_knnclassifier()
