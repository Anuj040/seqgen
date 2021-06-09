"""module for different (sequence, phone-number) generator functions"""

import os
import random
from typing import Iterator, Tuple, Union

import numpy as np
from PIL import Image

from utils.data_handler import DataLoader

# Common data loader object for all methods
LOADER = DataLoader("dataset")


def generate_numbers_sequence(
    digits: Union[Iterator[int]],
    image_width: Union[int, None] = None,
    spacing_range: Tuple[int, int] = (2, 10),
) -> np.ndarray:
    """method for generating an image for sequence of numbers

    Args:
        digits (Union[Iterator[int]]): Digits to be converted to a sequence
        image_width (Union[int, None]): Width of the final image.
        spacing_range (Tuple[int, int], optional): [description]. Defaults to (2, 10).

    Returns:
        np.ndarray: sequence-image array
    """

    # Default image height of MNIST digits
    image_height = 28

    # If no image width has been provided,
    assert image_width is not None, "Image width needs to be spedicfied"

    images = []
    for digit in digits:
        # Retrieve a random image of "digit"
        image = LOADER.retrieve(digit)
        images.append(image)

    # Generate a single sequence image from all digits
    img_seq = np.concatenate(images, axis=1)

    # Pillow object for resizing and saving
    pil_img = Image.fromarray(img_seq)
    pil_img_seq = pil_img.resize((image_width, image_height))

    # Color Inversion -> white background & black text
    # Normalized (0.0 - 1.0)
    return (255.0 - np.array(pil_img_seq, dtype=np.float32)) / 255.0


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
        images.append(generate_numbers_sequence(digits, image_width=image_width))

    # Save the image file
    if output_path is not None:
        print("=" * 23 + "\nSaving the image files.\n" + "=" * 23)
        os.makedirs(output_path, exist_ok=True)
        for i, sequence in enumerate(random_sequences):
            # Convert to Pillow object
            pil_img_seq = Image.fromarray((images[i] * 255.0).astype(np.uint8))
            # Filename and location for saving the image
            save_img_name = str(sequence)
            save_img_path = os.path.join(output_path, save_img_name)
            pil_img_seq.save(f"{save_img_path}.png")


if __name__ == "__main__":
    # img = generate_numbers_sequence(digits=[4, 1, 1], image_width=None)
    generate_phone_numbers(10, image_width=500, output_path="outputs")
