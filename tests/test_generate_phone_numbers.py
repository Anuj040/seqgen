"""module for unittests on generate_phone_numbers of nunmber_generator module"""
import glob
import os
import random
import shutil
import sys

import numpy as np
from PIL import Image

sys.path.append("./")
from number_generator.num_gen import generate_phone_numbers as gpn


def test_execution_details():
    """test the generated files from generate_phone_numbers are as expected"""

    output_path = "./tests/test_outputs"

    # check for different cases
    for n in range(2, 10):

        # Remove the existing folder
        if os.path.exists(output_path):
            shutil.rmtree(output_path)

        # Pick a random width for the output image
        image_width = random.randint(200, 500)
        gpn(n, image_width=image_width, output_path=output_path)

        # Check if the desired number of files was generated
        image_files = glob.glob(os.path.join(output_path, "*.png"))
        assert len(image_files) == n

        # Check the size of the image files is as desired
        image = Image.open(random.choice(image_files))
        assert np.array(image).shape == (28, image_width)
