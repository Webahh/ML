from model_input import add_empty_frames, ModelInput
from model import Model
import numpy as np
import atomics
import threading
import time


class AtomicAltar:
    """A thread save method to send a single piece of data between threads."""

    def __init__(self):
        """Initializes an empty atomic altar"""
        self._gate = atomics.atomic(width=4, atype=atomics.UINT)
        self._gate.store(0)
        self._altar = None

    def destroy(self):
        """Destroys the atomic altar"""
        self._gate.store(39)

    def is_destroyed(self):
        """Checks whether or not the altar is destroyed"""
        return self._gate.load() == 39

    def offering_available(self):
        """Checks whether or not there is an offering available to consume"""
        return self._gate.load() == 1

    def is_empty(self):
        """Checks whether or not the altar is empty"""
        return self._gate.load() == 0

    def offer(self, v):
        """Puts 'v' on the altar. Blocks until the altar is empty, before offering 'v'"""
        while self._gate.load() == 1:
            time.sleep(0.0001)

        self._altar = v
        self._gate.store(1)

    def consume(self):
        """Takes the offered value from the altar, blocking if no offering is available"""
        while self._gate.load() == 0:
            time.sleep(0.0001)

        val = self._altar
        self._altar = None
        self._gate.store(0)

        return val


class ModelInputBuffer:
    def __init__(
        self, model: Model, source_fps: float, callback: callable, target_fps=60
    ):
        self._target_fps = target_fps
        self._source_fps = source_fps
        self._upscale_t = target_fps // source_fps
        self._upscale_o = target_fps % source_fps
        self._overflow = self._upscale_o
        self._last_frame = None
        self._frame = 0
        self._last_infer = 0
        self._buffer = np.array(
            add_empty_frames([], model.sequence_length), dtype=np.int16
        )

        self._model = model
        self._altar = AtomicAltar()

        def infer_worker(altar: AtomicAltar):
            while True:
                if altar.is_destroyed():
                    break

                buffer = altar.consume()

                if buffer is None:
                    continue

                result = self._model.infer(buffer)
                callback(result)

        threading.Thread(target=lambda: infer_worker(self._altar), daemon=True).start()

    def push_all(self, inputs: [ModelInput]):
        for inp in inputs:
            self.push(inp)

    def push(self, input: ModelInput):
        inputs = [input]

        if self._last_frame is not None and self._upscale_t > 1:
            steps = max(0, self._upscale_t - 1)
            self._overflow += self._upscale_o

            if self._overflow > self._target_fps:
                steps = self._upscale_t + 1
                self._overflow = self._overflow % self._source_fps

            inputs = self._last_frame.interpolate_to(input, int(steps))

        self._last_frame = input

        # self._frame += 1
        frames = [f.flattened() for f in inputs]

        # Save the old buffer, in case it needs to be send to the model

        old_buffer = self._buffer

        # Ring buffer von kleinanzeigen
        self._buffer = np.insert(self._buffer, 0, frames, axis=0)
        self._buffer = np.delete(
            self._buffer,
            (
                max(0, self._model.sequence_length - len(frames)),
                self._model.sequence_length - 1,
            ),
            axis=0,
        )

        # Check if the model needs a new buffer and if at least two frames have passed
        if self._altar.is_empty():  # and self._frame - self._last_infer > 2:
            # self._last_infer = self._frame
            self._altar.offer(old_buffer)

    def destroy(self):
        self._altar.destroy()
