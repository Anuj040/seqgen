"""module for different (sequence, phone-number) generator functions"""

import os
import random
import sys
from typing import Iterator, Tuple, Union

import numpy as np
from PIL import Image

# pylint: disable = wrong-import-position, import-error, no-name-in-module
sys.path.append("./")
from number_generator.utils.data_handler import DataLoader

# Common data loader object for all methods
LOADER = DataLoader("dataset")


def generate_numbers_sequence(
    digits: Iterator[int],
    image_width: Union[int, None] = None,
    spacing_range: Tuple[int, int] = (2, 10),
    output_path: Union[None, str] = None,
) -> np.ndarray:
    """method for generating an image for sequence of numbers

    Args:
        digits (Iterator[int]): Digits to be converted to a sequence
        image_width (Union[int, None]): Width of the final image.
        spacing_range (Tuple[int, int], optional): [description]. Defaults to (2, 10).
        output_path (Union[None, str]): Path for saving image files.

    Returns:
        np.ndarray: sequence-image array
    """

    # Default image height of MNIST digits
    image_height = 28

    # If no image width has been provided,
    assert image_width is not None, "Image width needs to be specified"

    images = []
    save_img_name = ""
    for digit in digits:
        # Retrieve a random image of "digit"
        image = LOADER.retrieve(digit)
        images.append(image)
        # File name for saving image
        save_img_name += str(digit)

    # Generate a single sequence image from all digits
    img_seq = np.concatenate(images, axis=1)

    # Color Inversion -> white background & black text
    # Pillow object for resizing and saving
    pil_img = Image.fromarray(255 - img_seq)
    pil_img_seq = pil_img.resize((image_width, image_height))

    # Save the image file
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)

        # location for saving the image
        save_img_path = os.path.join(output_path, save_img_name)
        pil_img_seq.save(f"{save_img_path}.png")

    # Normalized (0.0 - 1.0)
    return (np.array(pil_img_seq, dtype=np.float32)) / 255.0


def generate_phone_numbers(
    num_images: int,
    image_width: Union[int, None] = None,
    spacing_range: Tuple[int, int] = (2, 10),
    output_path: Union[None, str] = None,
    seed: Union[int, None] = None,
):
    """method for generating images for random phone numbers

    Args:
        num_images (int): Number of images to be generated
        image_width (Union[int, None]): Width of the final image.
        spacing_range (Tuple[int, int], optional): [description]. Defaults to (2, 10).
        output_path (Union[None, str]): Path for saving image files.
        seed (Union[None, int]): seed for the random lib for reproducibility. Defaults to None
    """
    # If no image width has been provided,
    assert image_width is not None, "Image width needs to be spedicfied"
    random.seed(seed)

    # Empty set for storing unique sequences
    random_sequences = set()

    # Generate random, unique 10-digit sequences
    while len(random_sequences) < num_images:
        random_sequences.add(random.randint(1_000_000_000, 9_999_999_999))

    images = []
    for sequence in random_sequences:
        # Add leading zero to phone number like sequences
        sequence = "0" + str(sequence)
        # Build a iterator function for digits
        digits = (int(digit) for digit in sequence)
        # Sequence image arrays
        images.append(
            generate_numbers_sequence(
                digits, image_width=image_width, output_path=output_path
            )
        )

    # Save the image file
    if output_path is not None:
        str_length = 24 + len(output_path)
        print(
            "=" * str_length
            + f"\nImages are saved at '{output_path}/'.\n"
            + "=" * str_length
        )


if __name__ == "__main__":
    # img = generate_numbers_sequence(digits=[4, 1, 1], image_width=None)
    generate_phone_numbers(10, image_width=500, output_path="outputs")
