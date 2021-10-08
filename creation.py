import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm


def convert_to_binary(images):
    return images > np.repeat(images.mean(axis=(1, 2)), 28 * 28, axis=0).reshape((images.shape[0], 28, 28))


def dump(images, file_name):
    images.astype(np.bool).tofile(file_name)


def images_mask(images, shadow_percentage):
    k, h, w = images.shape
    c = np.zeros((k, 2))
    c[:, 0] = shadow_percentage
    c[:, 1] = 1 - shadow_percentage
    return np.asarray([np.random.choice([0, 1], h * w, p=cc).reshape((h, w)) for cc in c])


def shadow(images, shadow_percentage):
    mask = images_mask(images, shadow_percentage)
    return (mask & images), mask


def create_dataset(images, base='./data'):
    binary_images = convert_to_binary(images)

    for shadow_percentage in tqdm(np.arange(0.0, 1.01, 0.05)):
        new_images, mask = shadow(binary_images, shadow_percentage)

        path = f'{base}/shadow={shadow_percentage:.2f}'
        os.makedirs(path)
        dump(new_images, f'{path}/images.bin')
        dump(mask, f'{path}/mask.bin')


if __name__ == '__main__':
    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
    create_dataset(train_images, base='./data/train')
    create_dataset(test_images, base='./data/test')
