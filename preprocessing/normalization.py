
def norm(dataset):
    min_vals = dataset.min(0)
    max_vals = dataset.max(0)
    ranges = max_vals - min_vals
    norm_dataset = zeros(shape(dataset))
    m = dataset.shape[0]
    norm_dataset = dataset - tile(min_vals, (m, 1))
    norm_dataset = norm_dataset/tile(ranges, (m, 1))
    return norm_dataset, ranges, min_vals
