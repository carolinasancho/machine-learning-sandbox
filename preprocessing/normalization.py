import numpy as np

def norm(dataset):
    min_vals = dataset.min(0)
    max_vals = dataset.max(0)
    ranges = max_vals - min_vals
    norm_dataset = np.zeros(np.shape(dataset))
    m = dataset.shape[0]
    norm_dataset = dataset - np.tile(min_vals, (m, 1))
    norm_dataset = norm_dataset/np.tile(ranges, (m, 1))
    return norm_dataset, ranges, min_vals

def test_norm():
    a = np.array([[15.0,15.0,15.0],[10.0,10.0,10.0],[5.0,5.0,5.0],[1.0,1.0,1.0]])
    d, r, m = norm(a)
    print d

if __name__ == '__main__':
    test_norm()
