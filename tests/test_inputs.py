from dataclasses import replace
import unittest
import numpy as np

from ai.controller import NetworkController
from ai.inputs import FEATURE_NAMES, encode_inputs
from ai.neural_network import NeuralNetwork, PARAMETER_SHAPES
from config import DEFAULT_SETTINGS
from game.environment import Environment


class InputTests(unittest.TestCase):
    def test_five_features_have_documented_order_scale_and_sign(self) -> None:
        env = Environment(seed=42)
        env.bird.y = 330
        env.bird.velocity = 340
        env.pipes[0].x = 300
        env.pipes[0].gap_top = 220
        np.testing.assert_allclose(encode_inputs(env),
                                   [0, 0.5, 172 / 480, -110 / 660, 72 / 660])
        self.assertEqual(len(FEATURE_NAMES), 5)
        self.assertEqual(encode_inputs(env).shape, (5,))

    def test_clipping_and_nonfinite_observations(self) -> None:
        env = Environment(seed=0)
        env.bird.y = -100000
        env.bird.velocity = -100000
        env.pipes[0].x = 100000
        np.testing.assert_array_equal(encode_inputs(env), [-1, -1, 1, 1, 1])
        env.bird.y = float("nan")
        with self.assertRaises(ValueError):
            encode_inputs(env)

    def test_flap_normalizes_against_larger_velocity_magnitude(self) -> None:
        env = Environment(replace(DEFAULT_SETTINGS, flap_velocity=-1000), seed=0)
        env.bird.velocity = -1000
        self.assertEqual(encode_inputs(env)[1], -1)
        env.bird.velocity = 680
        self.assertAlmostEqual(encode_inputs(env)[1], 0.68)

    def test_uniformly_scaled_world_keeps_same_observation(self) -> None:
        scaled = replace(DEFAULT_SETTINGS, width=960, height=1440, ground_height=120,
                         bird_x=256, bird_radius=28, gravity=2800, flap_velocity=-840,
                         terminal_velocity=1360, pipe_width=144, pipe_gap=364,
                         pipe_spacing=490, pipe_speed=380, gap_margin=144, first_pipe_offset=200)
        a, b = Environment(seed=7), Environment(scaled, seed=7)
        a.bird.velocity, b.bird.velocity = 100, 200
        np.testing.assert_allclose(encode_inputs(a), encode_inputs(b))

    def test_target_is_current_pipe_until_whole_bird_clears_it(self) -> None:
        env = Environment(seed=0)
        first, second = env.pipes[:2]
        first.gap_top, second.gap_top = 200, 300
        first.x = env.bird.x - first.width - 8
        self.assertAlmostEqual(encode_inputs(env)[3], (200 - env.bird.y) / 660)
        first.x = env.bird.x - env.bird.radius - first.width - 1
        self.assertAlmostEqual(encode_inputs(env)[3], (300 - env.bird.y) / 660)

    def test_observations_do_not_change_environment_or_randomness(self) -> None:
        env = Environment(seed=0)
        rng_state = env.rng.getstate()
        before = (env.bird.y, env.bird.velocity, env.ticks, env.score, env.distance)
        encode_inputs(env)
        self.assertEqual(before, (env.bird.y, env.bird.velocity, env.ticks, env.score, env.distance))
        self.assertEqual(rng_state, env.rng.getstate())


class ControllerTests(unittest.TestCase):
    def test_network_drives_flap_and_no_flap_without_graphics(self) -> None:
        parameters = {name: np.zeros(shape) for name, shape in PARAMETER_SHAPES.items()}
        network = NeuralNetwork(parameters=parameters)
        controller = NetworkController(network)
        env = Environment(seed=0)
        env.step(controller(env))
        self.assertGreater(env.bird.velocity, 0)
        self.assertEqual(controller.last_output, 0.5)
        self.assertFalse(controller.last_action)
        self.assertEqual(controller.last_inputs.shape, (5,))
        network.b2[0] = 1
        env.step(controller(env))
        self.assertLess(env.bird.velocity, 0)
        self.assertGreater(controller.last_output, 0.5)
        self.assertTrue(controller.last_action)
        env.bird.alive = False
        self.assertFalse(controller(env))

    def test_same_network_and_environment_seed_replay_same_episode(self) -> None:
        a, b = Environment(seed=11), Environment(seed=11)
        controllers = [NetworkController(NeuralNetwork(seed=4)) for _ in range(2)]
        for _ in range(600):
            a.step(controllers[0](a))
            b.step(controllers[1](b))
            self.assertEqual((a.bird.y, a.bird.velocity, a.score, a.death_reason),
                             (b.bird.y, b.bird.velocity, b.score, b.death_reason))
            if a.done:
                break
        self.assertTrue(a.done)


if __name__ == "__main__":
    unittest.main()
