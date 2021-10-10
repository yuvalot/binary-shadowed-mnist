import numpy as np
import pandas as pd
from tqdm import tqdm


def create_indices(set_path):
    images = np.fromfile(f'{set_path}/shadow=0.00/images.bin', dtype=np.bool).reshape((60000, 28, 28))
    ones_counts = images.sum(axis=(1, 2))
    zeros_counts = (1 - images).sum(axis=(1, 2))
    ones_counts_expanded = np.tile(ones_counts, (28, 28, 1)).T
    zeros_counts_expanded = np.tile(zeros_counts, (28, 28, 1)).T

    sample_probabilities = \
        ((1 / (ones_counts_expanded * 2)) * images) + \
        ((1 / (zeros_counts_expanded * 2)) * (1 - images))

    sample_range = np.arange(28 ** 2)
    sample_locations = np.asarray(
        [np.random.choice(sample_range, size=1, p=p) for p in tqdm(sample_probabilities.reshape(-1, 784))])
    sample_values = np.take_along_axis(images.reshape(-1, 784), sample_locations, axis=1)
    indices = np.concatenate((sample_locations, sample_values), axis=1)
    pd.DataFrame(indices, columns=('index', 'value')).to_csv(f'{set_path}/indices.csv', index=False)


if __name__ == '__main__':
    create_indices(set_path='./data/train')
    create_indices(set_path='./data/train')
