import numpy as np


class Epsilon:
    def __init__(self, initial_epsilon=1.0, max_epsilon=1.0, min_epsilon=0.01, decay_rate=0.005):
        self._epsilon = initial_epsilon
        self._max = max_epsilon
        self._min = min_epsilon
        self._decay_rate = decay_rate

    def decay(self, i):
        self._epsilon = self._min + (self._max - self._min) * np.exp(-self._decay_rate * i)
        return self._epsilon

    @property
    def value(self):
        return self._epsilon
