"""module for data generator class for ML training"""
import sys
import warnings

import numpy as np

# pylint: disable = wrong-import-position, import-error
sys.path.append("./")
from number_generator.num_gen import generate_phone_numbers


class DataGenerator:
    """Data generator class for returing batches of input-output pair"""

    def __init__(
        self,
        batch_size: int = 32,
        image_width: int = 400,
        augment: bool = False,
        output_path: str = None,
    ) -> None:
        """
        Args:
            batch_size (int, optional): Defaults to 32.
            image_width (int, optional): Width of the images in the batch. Defaults to 400.
            augment (bool, optional): Apply image augmentations. Defaults to False.
            output_path (str, optional): Path for saving generated images. Use only for testing.
                                        Defaults to None.
        """
        self.batch_size = batch_size
        self.image_width = image_width
        self.augment = augment
        self.output_path = output_path
        if self.output_path is not None:
            warnings.warn(
                f"Generated image files will be saved at '{self.output_path}'. \
                Depending on the train length, it might take-up significant memory",
            )

    def __call__(self):
        while True:
            image_batch, sequence_batch = generate_phone_numbers(
                num_images=self.batch_size,
                image_width=self.image_width,
                augment=self.augment,
                verbose=False,
                output_path=self.output_path,
            )
            image_batch = np.concatenate(
                [np.expand_dims(image, axis=0) for image in image_batch], axis=0
            )
            sequence_batch = np.concatenate(
                [np.expand_dims(sequence, axis=0) for sequence in sequence_batch],
                axis=0,
            )
            yield np.expand_dims(image_batch, axis=-1), sequence_batch


if __name__ == "__main__":
    generator = DataGenerator(augment=True, output_path="outputs")
    for inputs, outputs in generator():
        print(outputs.shape)
