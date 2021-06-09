"""module for extracting, preporcessing, etc. of data"""
import gzip
import os
import random
import zipfile
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import requests


def training_images(path: str) -> np.ndarray:
    """method to extract the images from byte file

    Args:
        path (str): path to the byte file

    Returns:
        np.ndarray: Image Array (#images, height, width)
    """
    with gzip.open(path, "r") as file:

        # first 4 bytes is a magic number
        file.read(4)

        # second 4 bytes is the number of images
        image_count = int.from_bytes(file.read(4), "big")

        # third 4 bytes is the row count
        row_count = int.from_bytes(file.read(4), "big")

        # fourth 4 bytes is the column count
        column_count = int.from_bytes(file.read(4), "big")

        # rest is the image pixel data, each pixel is stored as an unsigned byte
        # pixel values are 0 to 255
        image_data = file.read()
        images = np.frombuffer(image_data, dtype=np.uint8).reshape(
            (image_count, row_count, column_count)
        )
        return images


def training_labels(path: str) -> np.ndarray:
    """method to extract image labels from byte file

    Args:
        path (str): path to the byte file

    Returns:
        np.ndarray: Image labels (0-9)
    """
    with gzip.open(path, "r") as file:
        # first 4 bytes is a magic number # Skip
        file.read(4)

        # second 4 bytes is the number of labels
        label_count = int.from_bytes(file.read(4), "big")

        # rest is the label data, each label is stored as unsigned byte
        # label values are 0 to 9
        label_data = file.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        return labels


def data_downloader(path: str) -> None:
    """Method for downloading the dataset if not available at the directed location

    Args:
        path (str): Directed location
    """
    url = "https://data.deepai.org/mnist.zip"
    r = requests.get(url, allow_redirects=False)

    mnist_path = os.path.join(path, "mnist.zip")
    with open(mnist_path, "wb") as f:

        # Saving received content as a zip file
        f.write(r.content)

    # Extrcting the contents of the downloaded file
    with zipfile.ZipFile(mnist_path, "r") as zip_ref:
        zip_ref.extractall(path)

    # Cleaning # Remove .zip file
    os.remove(mnist_path)


class DataLoader:
    """Class for retriving and sorting data"""

    def __init__(self, path: str) -> None:

        if not os.path.exists(path):
            # If the data does not exist at directed location
            # Download and extract the data
            os.makedirs(path)
            print("Data not available. Downloading it from the internet.")
            data_downloader(path)
            print("Download finished.")

        # retrieve image & labels objects
        images_path = os.path.join(path, "train-images-idx3-ubyte.gz")
        self.images = training_images(images_path)

        labels_path = os.path.join(path, "train-labels-idx1-ubyte.gz")
        self.labels = training_labels(labels_path)

        # Prepare a dict of image array indices for each digit
        self.digit_indices = {}
        for i in range(10):
            self.digit_indices[i] = np.where(self.labels == i)[0]

    def retrieve(self, digit: int, seed: Union[None, int] = None) -> np.ndarray:
        """method for picking a random image for "digit"

        Args:
            digit (int): digit to be retrieved
            seed (Union[None, int]): seed for the random lib for reproducibility. Defaults to None

        Returns:
            np.ndarray: image array
        """
        random.seed(seed)
        object_index = random.choice(self.digit_indices[digit])
        return self.images[object_index]


if __name__ == "__main__":
    loader = DataLoader("dataset")
    image = loader.retrieve(6, seed=123)
    plt.imshow(image)
    plt.show()
