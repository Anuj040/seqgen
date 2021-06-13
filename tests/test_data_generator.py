"""module for tests on data generator class"""
import random
import sys
import warnings

sys.path.append("./")

from number_generator.data_generator import DataGenerator


def test_generator_details():
    """test batches coming from data generator are as expected"""

    for _ in range(5):
        batch_size = random.randint(32, 128)
        image_width = random.randint(200, 500)
        generator = DataGenerator(batch_size=batch_size, image_width=image_width)
        for _ in range(10):
            inputs, outputs = next(generator())
            assert inputs.shape == (batch_size, 28, image_width)
            assert outputs.shape == (batch_size,)


def test_raise_warning():
    """test to ensure the relevant warnings are raised"""
    output_path = "./tests/test_outputs"
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        generator = DataGenerator(output_path=output_path)
        # Verify some things
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
