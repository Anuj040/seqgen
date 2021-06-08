"""module for extracting, preporcessing, etc. of data"""
import gzip
import os

import numpy as np


def training_images(path: str) -> np.ndarray:
    """method to extract the images from byte file

    Args:
        path (str): path to the byte file

    Returns:
        np.ndarray: Image Array (#images, height, width)
    """
    with gzip.open(path, "r") as f:

        # first 4 bytes is a magic number
        f.read(4)

        # second 4 bytes is the number of images
        image_count = int.from_bytes(f.read(4), "big")

        # third 4 bytes is the row count
        row_count = int.from_bytes(f.read(4), "big")

        # fourth 4 bytes is the column count
        column_count = int.from_bytes(f.read(4), "big")

        # rest is the image pixel data, each pixel is stored as an unsigned byte
        # pixel values are 0 to 255
        image_data = f.read()
        images = (
            np.frombuffer(image_data, dtype=np.uint8)
            .reshape((image_count, row_count, column_count))
            .astype(np.float32)
        )
        # Color Inversion -> white background & black text
        return 255.0 - images


def training_labels(path: str) -> np.ndarray:
    """method to extract image labels from byte file

    Args:
        path (str): path to the byte file

    Returns:
        np.ndarray: Image labels (0-9)
    """
    with gzip.open(path, "r") as f:
        # first 4 bytes is a magic number # Skip
        f.read(4)

        # second 4 bytes is the number of labels
        label_count = int.from_bytes(f.read(4), "big")

        # rest is the label data, each label is stored as unsigned byte
        # label values are 0 to 9
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        return labels


class DataLoader:
    def __init__(self, path: str) -> None:

        # retrieve image & labels objects
        images_path = os.path.join(path, "train-images-idx3-ubyte.gz")
        images = training_images(images_path)

        labels_path = os.path.join(path, "train-labels-idx1-ubyte.gz")
        labels = training_labels(labels_path)


if __name__ == "__main__":
    loader = DataLoader("dataset")
