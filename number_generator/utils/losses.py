"""common module for all loss and metric definitions"""
import tensorflow as tf


class SequenceAccuracy(tf.keras.metrics.Metric):
    """custom keras metric layer to calculate sequence accuracy"""

    def __init__(self, name: str = "acc", **kwargs) -> None:
        """
        Args:
            name (str, optional): Layer name. Defaults to "acc".
        """
        super().__init__(name=name, **kwargs)

        # Stores the summation of metric value over the whole dataset
        self.metric = self.add_weight(name="true_count", initializer="zeros")

        # Samples count
        self.metric_count = self.add_weight(name="Count", initializer="zeros")

    # pylint: disable = arguments-differ
    def update_state(self, y_true, y_pred, **kwargs) -> None:

        correct_pred = tf.cast(y_true, y_pred.dtype) == y_pred
        correct_pred = tf.cast(correct_pred, tf.float32)

        # Number of samples in a given batch
        count = tf.cast(tf.shape(correct_pred), self.dtype)
        # Sum of metric value for the processed samples
        self.metric.assign_add(tf.reduce_sum(correct_pred))
        # Total number of samples processed
        self.metric_count.assign_add(tf.reduce_prod(count))

    def result(self) -> tf.Tensor:
        # Average metric value
        return self.metric / self.metric_count

    def reset_states(self):
        # metric state reset at the start of each epoch.
        self.metric.assign(0.0)
        self.metric_count.assign(0)
