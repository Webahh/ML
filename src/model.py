import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import GRU, Dense, Input
from tensorflow.keras.models import Sequential
from model_input import add_empty_frames, flatten_frames, load_training_data


class Model:
    def __init__(self, training_data=None):
        if training_data is None:
            training_data = load_training_data("ressources/gestures")

        # Find longest sequence of frames for one gesture
        self._length = 0
        for _, frames in training_data:
            if len(frames) > self._length:
                self._length = len(frames)

        # Build a dictionary over all labels
        self._labels = {}
        self._label_count = 0
        for label, _ in training_data:
            if label not in self._labels:
                self._labels[label] = self._label_count
                self._label_count += 1

        # Build inverted label dict
        self._lables_inv = {index: label for label, index in self._labels.items()}

        # Get training data labels
        training_labels = [self._labels[label] for label, _ in training_data]
        for i, v in enumerate(training_labels):
            training_labels[i] = [0] * self._label_count
            training_labels[i][v] = 1.0

        training_labels = np.array(training_labels)

        # Make all training data seuqnces the same length (number of frames) and flatten each frame
        training_inputs = [flatten_frames(seq) for _, seq in training_data]
        training_inputs = [
            add_empty_frames(seq, self._length) for seq in training_inputs
        ]

        # Assert input dimensions
        assert len(training_inputs) == len(training_labels)
        assert len(training_inputs[0]) == self._length
        assert len(training_inputs[0][0]) == 2 * 22 * 3

        # Convert to ndarray and assert shape
        training_inputs = np.array(training_inputs, dtype=np.int16)
        assert training_inputs.shape == (len(training_labels), self._length, 2 * 22 * 3)

        self._model = Sequential(
            [
                Input(shape=(self._length, 2 * 22 * 3)),
                GRU(
                    128,
                    activation="tanh",
                    return_sequences=True,
                ),
                GRU(64, activation="tanh"),
                Dense(self._label_count, activation="sigmoid"),
            ]
        )

        self._model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        self._model.summary()
        self._model.fit(training_inputs, training_labels, epochs=200, batch_size=8)

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

        for output, conf in enumerate(outputs):
            conf = np.max(conf)
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
    model = Model()
    model.save()

    loaded_model = Model.load()
    assert model._model is not None
    assert loaded_model._model is not None
