import numpy as np
import operator


## TODO: features weight?
## TODO: add other distances measures (manhattan)

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

class KNNRegressor(KNN):

    def __init__(self, neighbors=5):
        self.neighbors = neighbors

    def predict(self, input, dataset, target):
        sorted_distance_indices = self.nearest_neighbors(input, dataset)
        prediction = np.zeros(self.neighbors)
        for i in range(self.neighbors):
            prediction[i] = target[sorted_distance_indices[i]]
        return np.mean(prediction)



def test_knnclassifier():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    knn = KNNClassifier(3)
    print knn.classify([0,0], group, labels)

def test_knnregressor():
    train = np.array([[1.0,6.0], [2.0,4.0], [3.0,7.0], [6.0,8.0], [7.0,1.0], [8.0,4.0]])
    target = [7.0, 8.0, 16.0, 44.0, 50.0, 68.0]
    knn = KNNRegressor(3)
    print knn.predict([4.0,2.0], train, target)

if __name__ == '__main__':
    test_knnclassifier()
    test_knnregressor()
