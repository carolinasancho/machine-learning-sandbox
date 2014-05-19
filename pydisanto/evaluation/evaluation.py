import numpy

# Confusion Matrix for classes enumarated from 0 to number of classes - 1
def confusion_matrix(predicted, target, number_of_classes):
    matrix = numpy.zeros((number_of_classes, number_of_classes))
    for yp, yt in zip(predicted,target):
        matrix[yp[0]][yt[0]] += 1
    return matrix


