"""main module for defining modeule related methods"""

import os
import sys

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM

# pylint: disable = wrong-import-position
sys.path.append("./")
from number_generator.data_generator import DataGenerator
from number_generator.utils.losses import SequenceAccuracy


class SequenceModel:
    """Sequence generator model class"""

    def __init__(
        self,
        model_dir: str,
        sequence_length: int = 11,
        initial_epoch: int = 0,
    ) -> None:
        """Inititalization method

        Args:
            sequence_length (int, optional): Sequence to be predicted. Defaults to 11.
            model_dir (str, optional): Directory for saving models.
            initial_epoch (int, optional): identifier for loading model. Defaults to 0.
        """

        self.initial_epoch = initial_epoch
        self.model_dir = model_dir
        self.model = (
            self.build(sequence_length=sequence_length)
            if not self.initial_epoch
            else KM.load_model(
                f"{self.model_dir}/model_{initial_epoch}.h5",
                custom_objects={"SequenceAccuracy": SequenceAccuracy()},
            )
        )

    def build(self, sequence_length: int = 11) -> KM.Model:
        """Method to build the model architecture

        Args:
            sequence_length (int, optional): Defaults to 11.

        Returns:
            KM.Model: sequence generator model
        """

        input_tensor = KL.Input(shape=(28, 400, 1))

        # Feature extractor
        conv_filters = KL.Conv2D(4, 3, (2, 2))(input_tensor)
        conv_filters = KL.BatchNormalization()(conv_filters)
        conv_filters = KL.Activation("relu")(conv_filters)
        conv_filters = KL.Conv2D(16, 3, (2, 2))(conv_filters)
        conv_filters = KL.BatchNormalization()(conv_filters)
        conv_filters = KL.Activation("relu")(conv_filters)

        features = KL.Flatten()(conv_filters)
        features = KL.Dense(2048)(features)
        features = KL.BatchNormalization()(features)
        features = KL.Activation("relu")(features)
        features = KL.Dense(512, activation="relu")(features)
        features = tf.expand_dims(features, axis=1)

        # Sequence Generator
        outputs = KL.LSTM(sequence_length, activation="relu")(features)
        outputs = tf.clip_by_value(outputs, 0.0, 10.0 - 1e-5)  # outputs = [0, 10)
        return KM.Model(input_tensor, outputs)

    def compile(self):
        """compiles the model object with relevant settings"""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.mse,
            metrics=SequenceAccuracy(),
        )

    def train(self, epochs: int):
        """method for execute model training

        Args:
            epochs (int): training epochs
        """
        # Data generator function
        generator = DataGenerator(augment=True)

        if not self.initial_epoch:
            self.compile()

        self.model.fit(
            generator(),
            initial_epoch=self.initial_epoch,
            epochs=epochs,
            steps_per_epoch=100,
            verbose=2,
            workers=1,
        )
        os.makedirs(f"{self.model_dir}", exist_ok=True)
        self.model.save(f"{self.model_dir}/model_{epochs}.h5")

    def eval(self):
        """method for model testing"""
        test_generator = DataGenerator(seed=123)
        print(self.model.evaluate(test_generator(), steps=1, verbose=0))

        # Sample examples
        inputs, outputs = next(test_generator())
        model_outputs = self.model(inputs).numpy().astype(np.uint8)
        for i in range(2):
            print("gt:", outputs[i])
            print("pr:", model_outputs[i])


if __name__ == "__main__":
    INITIAL_EPOCH = 0
    MODEL_DIR = "lstm_fconv_2048_512_nbat"

    EPOCHS = 128 if not INITIAL_EPOCH else 2 * INITIAL_EPOCH
    seq_model = SequenceModel(model_dir=MODEL_DIR, initial_epoch=INITIAL_EPOCH)

    seq_model.train(epochs=EPOCHS)
    seq_model.eval()
