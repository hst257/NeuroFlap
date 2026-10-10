"""Bridge the NumPy network to a simulation; no dependency on Pygame."""

from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
from .inputs import encode_inputs
from .neural_network import NeuralNetwork

if TYPE_CHECKING:
    from game.environment import Environment


class NetworkController:
    def __init__(self, network: NeuralNetwork) -> None:
        self.network = network
        self.last_inputs: NDArray[np.float64] | None = None
        self.last_output: float | None = None
        self.last_action = False

    def __call__(self, env: "Environment") -> bool:
        if env.done:
            self.last_action = False
            return False
        self.last_inputs = encode_inputs(env)
        self.last_output = self.network.forward(self.last_inputs)
        self.last_action = self.last_output > 0.5
        return self.last_action
