from math import log
from collections import defaultdict
import operator

def calcShannonEntropy(dataset):
    num_entries = len(dataset)
    labels_counts = defaultdict(int)
    for instance in dataset:
        labels_counts[instance[-1]] += 1
    shannonEntropy = 0.0
    for label in labels_counts:
        prob = float(labels_counts[label])/num_entries
        shannonEntropy -= prob*log(prob,2)
    return shannonEntropy

# return a new dataset conditional to the axis and value
def splitDataset(dataset, axis, value):
    ret_dataset = []
    for instance in dataset:
        if instance[axis] == value:
            reduced_instance = instance[:axis]
            reduced_instance.extend(instance[axis+1:])
            ret_dataset.append(reduced_instance)
    return ret_dataset

def chooseBestFeatureToSplit(dataset):
    num_features = len(dataset[0]) - 1
    base_entropy = calcShannonEntropy(dataset)
    best_info_gain = 0.0
    best_feature = -1
    for i in xrange(num_features):
        feature_list = [instance[i] for instance in dataset]
        unique_vals = set(feature_list)
        new_entropy = 0.0
        for value in unique_vals:
            sub_dataset = splitDataset(dataset, i, value)
            prob = len(sub_dataset)/float(len(dataset))
            new_entropy += prob*calcShannonEntropy(sub_dataset)
        info_gain = base_entropy - new_entropy
        if (info_gain > best_info_gain):
            best_info_gain = info_gain
            best_feature = i
    return best_feature

def majorityCount(classList):
    class_count = defaultdict(int)
    for vote in classList:
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]

def createTree(dataset, labels):
    class_list = [instance[-1] for instance in dataset]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(dataset[0]) == 1:
        return majorityCount(class_list)
    best_feature = chooseBestFeatureToSplit(dataset)
    best_feature_label = labels[best_feature]
    my_tree = {best_feature_label:{}}
    del(labels[best_feature])
    feature_values = [example[best_feature] for example in dataset]
    unique_vals = set(feature_values)
    for value in unique_vals:
        sub_labels = labels[:]
        my_tree[best_feature_label][value] = createTree(splitDataset(dataset, best_feature, value), sub_labels)
    return my_tree

def test_splitDataset():
    dataset = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    print 'Original dataset:'
    print dataset
    print 'Spliting o feature 0 and value 1:'
    print splitDataset(dataset, 0, 1)

def test_calcShannonEntropy():
    dataset = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    print calcShannonEntropy(dataset)
    print 'Effect of adding one more class to the entropy:'
    dataset[0][-1] = 'maybe'
    print calcShannonEntropy(dataset)

def test_chooseBestFeatureToSplit():
    dataset = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    print 'Best feature to split:'
    print chooseBestFeatureToSplit(dataset)

if __name__ == '__main__':
    #test_calcShannonEntropy()
    #test_splitDataset()
    test_chooseBestFeatureToSplit()
