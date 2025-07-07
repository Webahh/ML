import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Input,
)
from tensorflow.keras.models import Sequential

from model_input import load_training_data


class Model:
    def __init__(self, training_data=None):
        if training_data is None:
            training_data = load_training_data("ressources/gestures")

        # Build a dictionary over all labels
        self._labels = {}
        self._label_count = 0
        for label, _ in training_data:
            if label not in self._labels:
                self._labels[label] = self._label_count
                self._label_count += 1

        # Build inverted label dict
        self._lables_inv = {index: label for label, index in self._labels.items()}

        training_dict = {
            self._labels[label]: [gesture for gesture in gestures]
            for label, gestures in training_data
        }

        training_label_indices = []
        training_inputs = []

        for label, inputs in training_dict.items():
            for input in inputs:
                training_label_indices.append(label)
                training_inputs.append(input.flattened())

        training_labels = []
        for li in training_label_indices:
            output = [0.0] * self.label_count
            output[li] = 1.0
            training_labels.append(output)

        training_inputs = np.array(training_inputs)
        training_labels = np.array(training_labels)

        self._model = Sequential(
            [
                Input(shape=(2 * 22 * 3, 1)),
                Flatten(),
                Dense(2 * 22 * 3, activation="relu"),
                Dropout(0.2),
                Dense(128, activation="relu"),
                Dropout(0.2),
                Dense(128, activation="relu"),
                Dropout(0.2),
                Dense(self._label_count, activation="softmax"),
            ]
        )

        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        self._model.summary()
        self._model.fit(training_inputs, training_labels, epochs=80)

    @property
    def sequence_length(self) -> int:
        return self._length

    @property
    def label_count(self) -> int:
        return self._label_count

    def label_from_index(self, index: int) -> str:
        return self._lables_inv[index]

    def index_from_label(self, label: str) -> int:
        return self._labels[label]

    def save(self, dir="model"):
        pickle_path = os.path.join(dir, "class.pkl")
        model_path = os.path.join(dir, "model.keras")
        os.makedirs(dir, exist_ok=True)

        # Save the entire model outside of pickle
        self._model.save(model_path)

        # Save this class using pickle, but dont save the model in pickle
        model = self._model
        self._model = None

        with open(pickle_path, "wb") as f:
            pickle.dump(self, f)

        self._model = model

    def infer(self, buffer) -> (str, float):
        outputs = self._model.predict(np.array([buffer], dtype=np.int16), verbose=0)
        index, confidence = 0, 0.0

        for output, conf in enumerate(outputs[0]):
            if conf > confidence:
                confidence = conf
                index = output

        return self.label_from_index(index), confidence

    @staticmethod
    def load(dir="model"):
        pickle_path = os.path.join(dir, "class.pkl")
        model_path = os.path.join(dir, "model.keras")

        # Read this class from pickle
        me = None
        with open(pickle_path, "rb") as f:
            me = pickle.load(f)

        # Load model from keras file
        model = tf.keras.models.load_model(model_path)
        me._model = model

        # Return instance
        return me


# If this file is not imported as a module, train and save model
if __name__ == "__main__":
    print("Generating new model")
    model = Model()
    model.save()

    loaded_model = Model.load()
    assert model._model is not None
    assert loaded_model._model is not None
