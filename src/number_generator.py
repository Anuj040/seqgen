"""module for different (sequence, phone-number) generator functions"""

from typing import List, Tuple, Union

import numpy as np
from PIL import Image

from utils.data_handler import DataLoader

# Common data loader object for all methods
LOADER = DataLoader("dataset")


def generate_numbers_sequence(
    digits: Union[List[int], Tuple[int]],
    image_width: Union[int, None] = None,
    spacing_range: Tuple[int, int] = (2, 10),
) -> np.ndarray:
    """method for generating an image for sequence of numbers

    Args:
        digits (Union[List[int], Tuple[int]]): Digits to be converted to a sequence
        image_width (Union[int, None]): Width of the final image. None will use dafault value
        spacing_range (Tuple[int, int], optional): [description]. Defaults to (2, 10).

    Returns:
        np.ndarray: sequence-image array
    """

    # Default image height of MNIST digits
    image_height = 28

    # If no image width has been provided,
    # keep width of each individual image unchanged
    if image_width is None:
        image_width = image_height * len(digits)

    images = []
    for digit in digits:
        # Retrieve a random image of "digit"
        image = LOADER.retrieve(digit)
        images.append(image)

    # Generate a single sequence image from all digits
    img_seq = np.concatenate(images, axis=1)

    # Pillow object for resizing and saving
    pil_img = Image.fromarray(img_seq.astype(np.uint8), "L")
    pil_img_seq = pil_img.resize((image_width, image_height))

    return np.array(pil_img_seq, dtype=np.float32)


if __name__ == "__main__":
    img = generate_numbers_sequence(digits=[4, 1, 1], image_width=None)
    print(type(img))
