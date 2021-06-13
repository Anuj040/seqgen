"""module for different (sequence, phone-number) generator functions"""

import math
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


def scale_and_rotate_image(im: Image.Image) -> Image.Image:
    """Affine Image Augmentation
        Code from https://stackoverflow.com/a/49468270

    Args:
        im (Image.Image): image array

    Returns:
        Image.Image: augmented image array
    """
    # pylint: disable = invalid-name, too-many-locals

    # Rotation and scale factor for the affine transformations
    deg_ccw = random.randint(-15, 15)
    scale_x = random.random() + 0.5
    scale_y = random.random() + 0.5

    # Get image dimensions
    width, height = im.size
    angle = math.radians(-deg_ccw)

    cos_theta = math.cos(angle)
    sin_theta = math.sin(angle)

    scaled_w, scaled_h = width * scale_x, height * scale_y

    new_w = int(
        math.ceil(math.fabs(cos_theta * scaled_w) + math.fabs(sin_theta * scaled_h))
    )
    new_h = int(
        math.ceil(math.fabs(sin_theta * scaled_w) + math.fabs(cos_theta * scaled_h))
    )

    org_center_x = width / 2.0
    org_center_y = height / 2.0
    trans_center_x = new_w / 2.0
    trans_center_y = new_h / 2.0

    a = cos_theta / scale_x
    b = sin_theta / scale_x
    c = org_center_x - trans_center_x * a - trans_center_y * b
    d = -sin_theta / scale_y
    e = cos_theta / scale_y
    f = org_center_y - trans_center_x * d - trans_center_y * e
    im = im.transform(
        (new_w, new_h), Image.AFFINE, (a, b, c, d, e, f), resample=Image.BILINEAR
    )

    return im.resize((width, height))


def s_and_p_noise(image: np.ndarray) -> np.ndarray:
    """Introduce salt and pepper noise to the image

    Args:
        image (np.ndarray): an image numpy array

    Returns:
        np.ndarray: a numpy array of noise added image
    """

    # Proportion of noisy pixels
    noise_amount = 0.01 * random.random()
    # Proportion of the noisy pixels with salt noise
    s_vs_p = 0.8

    # Salt mode
    num_salt = np.ceil(noise_amount * image.size * s_vs_p)
    noisy_coords = tuple(np.random.randint(0, j, int(num_salt)) for j in image.shape)
    image[noisy_coords] = 255

    # Pepper mode
    num_pepper = np.ceil(noise_amount * image.size * (1.0 - s_vs_p))
    noisy_coords = tuple(np.random.randint(0, j, int(num_pepper)) for j in image.shape)
    image[noisy_coords] = 0

    return image


def generate_numbers_sequence(
    digits: Iterator[int],
    image_width: Union[int, None] = None,
    spacing_range: Tuple[int, int] = (2, 10),
    output_path: Union[None, str] = None,
    augment: bool = False,
) -> np.ndarray:
    """method for generating an image for sequence of numbers

    Args:
        digits (Iterator[int]): Digits to be converted to a sequence
        image_width (Union[int, None]): Width of the final image.
        spacing_range (Tuple[int, int], optional): [description]. Defaults to (2, 10).
        output_path (Union[None, str]): Path for saving image files.
        augment (bool, optional): Apply image augmentation

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

        if augment:
            # Pass a Pillow Image object
            image = Image.fromarray(image)

            # Apply single digit augmentation
            image = scale_and_rotate_image(image)

            # Convert to numpy array
            image = np.array(image)

        images.append(image)
        # File name for saving image
        save_img_name += str(digit)

    # Generate a single sequence image from all digits
    img_seq = np.concatenate(images, axis=1)

    # Whole seq augmentation
    if augment:

        img_seq = s_and_p_noise(img_seq)

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


# pylint: disable = too-many-arguments
def generate_phone_numbers(
    num_images: int,
    image_width: Union[int, None] = None,
    spacing_range: Tuple[int, int] = (2, 10),
    output_path: Union[None, str] = None,
    augment: bool = False,
    seed: Union[int, None] = None,
    verbose: bool = True,
):
    """method for generating images for random phone numbers

    Args:
        num_images (int): Number of images to be generated
        image_width (Union[int, None]): Width of the final image.
        spacing_range (Tuple[int, int], optional): [description]. Defaults to (2, 10).
        output_path (Union[None, str]): Path for saving image files.
        augment (bool, optional): Apply image augmentation
        seed (Union[None, int]): seed for the random lib for reproducibility. Defaults to None
        verbose (bool, optional): print messages
    """
    # If no image width has been provided,
    assert image_width is not None, "Image width needs to be specified"
    random.seed(seed)

    # Empty set for storing unique sequences
    random_sequences = set()

    # Generate random, unique 10-digit sequences
    while len(random_sequences) < num_images:
        random_sequences.add(random.randint(1_000_000_000, 9_999_999_999))

    images = []
    zero_prefix_sequences = []
    for sequence in random_sequences:
        # Add leading zero to phone number like sequences
        sequence = "0" + str(sequence)
        # Build a iterator function for digits
        digits = tuple(int(digit) for digit in sequence)
        # Sequence image arrays
        images.append(
            generate_numbers_sequence(
                digits,
                image_width=image_width,
                output_path=output_path,
                augment=augment,
            )
        )
        zero_prefix_sequences.append(digits)

    # Save the image file
    if verbose:
        if output_path is not None:
            str_length = 24 + len(output_path)
            print(
                "=" * str_length
                + f"\nImages are saved at '{output_path}/'.\n"
                + "=" * str_length
            )
        else:
            str_length = 55
            print(
                "=" * str_length
                + "\nOutput path not provided, skip saving generated images.\n"
                + "=" * str_length
            )
    return images, zero_prefix_sequences


if __name__ == "__main__":
    # img = generate_numbers_sequence(digits=[4, 1, 1], image_width=None)
    generate_phone_numbers(10, image_width=500, output_path="outputs", augment=True)
