"""The versioned five-feature contract shared by controllers and saved models."""

from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from game.environment import Environment

INPUT_SCHEMA = "neuroflap.inputs.v1"
FEATURE_NAMES = (
    "bird_y", "bird_velocity", "pipe_distance", "gap_top_delta", "gap_bottom_delta",
)


def encode_inputs(env: "Environment") -> NDArray[np.float64]:
    """Return five bounded, signed inputs from the current bird and next pipe.

    Screen y increases downward. Gap deltas are edge_y minus bird_y: a negative
    delta means that edge is above the bird. The current pipe remains the target
    until the entire bird clears it, matching the environment's scoring rule.
    """
    bird, pipe, s = env.bird, env.next_pipe, env.settings
    velocity_scale = max(abs(s.flap_velocity), s.terminal_velocity)
    values = np.array([
        2.0 * bird.y / s.play_height - 1.0,
        bird.velocity / velocity_scale,
        (pipe.x - bird.x) / s.width,
        (pipe.gap_top - bird.y) / s.play_height,
        (pipe.gap_bottom - bird.y) / s.play_height,
    ], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("Game observations must be finite")
    return np.clip(values, -1.0, 1.0)
