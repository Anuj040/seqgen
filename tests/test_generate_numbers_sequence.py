"""module for unittests on generate_numbers_sequence of nunmber_generator module"""
import os
import random
import shutil
import sys

import numpy as np

sys.path.append("./")
# Remove the existing dataset
shutil.rmtree("dataset")
from number_generator.num_gen import generate_numbers_sequence as gns


def test_output_details():
    """test the outputs of the generate_numbers_sequence method are as expected"""
    output_path = "./tests/test_outputs"

    # check for different cases
    for n in range(2, 10):

        # Remove the existing folder
        if os.path.exists(output_path):
            shutil.rmtree(output_path)

        # Generate a list of "n" digits
        digits = random.sample(range(0, 10), n)
        image_width = random.randint(100, 150)

        # Retrieve sequence image array
        image = gns(
            digits, image_width=image_width, output_path=output_path, augment=True
        )

        # Check for the shape and dtype of the output array
        assert image.shape == (28, image_width)
        assert image.dtype == np.float32
        # Check the range of values from the output array
        assert np.max(image) <= 1.0
        assert np.min(image) >= 0.0
