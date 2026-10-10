"""Independent numeric checks for forward propagation and model persistence."""

import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from ai.inputs import INPUT_SCHEMA
from ai.neural_network import FORMAT_VERSION, PARAMETER_SHAPES, NeuralNetwork, sigmoid


def zero_parameters() -> dict[str, np.ndarray]:
    return {name: np.zeros(shape) for name, shape in PARAMETER_SHAPES.items()}


class NeuralNetworkTests(unittest.TestCase):
    def test_architecture_and_initialization_are_reproducible(self) -> None:
        a, b, c = NeuralNetwork(seed=42), NeuralNetwork(seed=42), NeuralNetwork(seed=43)
        for name, shape in PARAMETER_SHAPES.items():
            value = getattr(a, name)
            self.assertEqual(value.shape, shape)
            self.assertEqual(value.dtype, np.float64)
            np.testing.assert_array_equal(value, getattr(b, name))
            self.assertTrue(np.isfinite(value).all())
        self.assertFalse(np.array_equal(a.w1, c.w1))
        np.testing.assert_array_equal(a.b1, np.zeros(8))
        np.testing.assert_array_equal(a.b2, np.zeros(1))
        self.assertEqual(sum(p.size for p in a.to_parameters().values()), 57)

    def test_zero_parameters_produce_half_and_do_not_flap(self) -> None:
        network = NeuralNetwork(parameters=zero_parameters())
        self.assertEqual(network.forward([1, -1, 0.5, 0, -0.1]), 0.5)
        self.assertIs(network.should_flap(np.zeros(5)), False)

    def test_known_one_neuron_calculation(self) -> None:
        parameters = zero_parameters()
        parameters["w1"][0, 0] = 2.0
        parameters["b1"][0] = -0.5
        parameters["w2"][0, 0] = 3.0
        parameters["b2"][0] = 0.1
        network = NeuralNetwork(parameters=parameters)
        expected = 1 / (1 + math.exp(-(3 * math.tanh(0.5) + 0.1)))
        self.assertAlmostEqual(network.forward([0.5, 0, 0, 0, 0]), expected, places=14)

    def test_full_forward_matches_independent_scalar_sums(self) -> None:
        network = NeuralNetwork(seed=8)
        rng = np.random.default_rng(12)
        for _ in range(20):
            inputs = rng.uniform(-1, 1, 5)
            hidden = [math.tanh(math.fsum(float(network.w1[j, i] * inputs[i])
                                          for i in range(5)) + float(network.b1[j]))
                      for j in range(8)]
            logit = math.fsum(float(network.w2[0, j]) * hidden[j] for j in range(8)) + float(network.b2[0])
            expected = 1 / (1 + math.exp(-logit))
            self.assertAlmostEqual(network.forward(inputs), expected, places=14)

    def test_bias_changes_decision_without_changing_weights(self) -> None:
        parameters = zero_parameters()
        parameters["b2"][0] = math.log(9)
        network = NeuralNetwork(parameters=parameters)
        self.assertAlmostEqual(network.forward(np.zeros(5)), 0.9)
        self.assertTrue(network.should_flap(np.zeros(5)))
        network.b2[0] = -math.log(9)
        self.assertAlmostEqual(network.forward(np.zeros(5)), 0.1)
        self.assertFalse(network.should_flap(np.zeros(5)))

    def test_sigmoid_is_stable_at_extreme_logits(self) -> None:
        with np.errstate(over="raise", invalid="raise"):
            result = sigmoid([-1000, -2, 0, 2, 1000])
        self.assertEqual(result[0], 0)
        self.assertEqual(result[2], 0.5)
        self.assertEqual(result[-1], 1)
        self.assertTrue(np.all(np.diff(result) > 0))
        self.assertAlmostEqual(result[1] + result[3], 1)

    def test_reject_invalid_inputs(self) -> None:
        network = NeuralNetwork(seed=0)
        for inputs in ([0] * 4, np.zeros((1, 5)), [0, 0, 0, 0, np.nan],
                       [0, 0, 0, 0, np.inf], [2, 0, 0, 0, 0], ["0"] * 5,
                       np.zeros(5, dtype=complex)):
            with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                network.forward(inputs)
        with self.assertRaises(ValueError):
            sigmoid([np.nan])

    def test_reject_invalid_parameter_shapes_types_and_values(self) -> None:
        for value in (np.zeros((5, 8)), np.full((8, 5), np.nan),
                      np.full((8, 5), np.inf), np.zeros((8, 5), dtype=complex)):
            parameters = zero_parameters()
            parameters["w1"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                NeuralNetwork(parameters=parameters)
        with self.assertRaises(ValueError):
            NeuralNetwork(parameters={"w1": np.zeros((8, 5))})

    def test_parameter_import_and_export_do_not_alias(self) -> None:
        parameters = zero_parameters()
        network = NeuralNetwork(parameters=parameters)
        parameters["w1"][0, 0] = 100
        self.assertEqual(network.w1[0, 0], 0)
        exported = network.to_parameters()
        exported["b2"][0] = 100
        self.assertEqual(network.b2[0], 0)

    def test_finite_but_overflowing_parameters_report_clear_error(self) -> None:
        parameters = zero_parameters()
        parameters["w1"][:] = np.finfo(np.float64).max
        with self.assertRaisesRegex(ValueError, "overflow"):
            NeuralNetwork(parameters=parameters).forward(np.ones(5))


class PersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "model.npz"

    def write_archive(self, **overrides: np.ndarray) -> None:
        fields = dict(NeuralNetwork(seed=0).to_parameters(),
                      format_version=np.array(FORMAT_VERSION),
                      architecture=np.array([5, 8, 1]), input_schema=np.array(INPUT_SCHEMA))
        fields.update(overrides)
        np.savez(self.path, **fields)

    def test_round_trip_preserves_every_parameter_and_output_exactly(self) -> None:
        network = NeuralNetwork(seed=5)
        network.b1[:] = np.arange(8) / 20
        network.b2[:] = 0.123
        path = self.path.parent / "nested" / "model.npz"
        network.save(path)
        loaded = NeuralNetwork.load(path)
        for name in PARAMETER_SHAPES:
            np.testing.assert_array_equal(getattr(network, name), getattr(loaded, name))
        inputs = [0.2, -0.6, 0.7, -0.1, 0.3]
        self.assertEqual(network.forward(inputs), loaded.forward(inputs))

    def test_metadata_is_stored_without_pickle(self) -> None:
        NeuralNetwork(seed=0).save(self.path)
        with np.load(self.path, allow_pickle=False) as archive:
            self.assertEqual(archive["format_version"].item(), 1)
            self.assertEqual(archive["input_schema"].item(), INPUT_SCHEMA)
            np.testing.assert_array_equal(archive["architecture"], [5, 8, 1])

    def test_reject_incompatible_and_malformed_archives(self) -> None:
        for fields in ({"format_version": np.array(999)},
                       {"architecture": np.array([5, 10, 1])},
                       {"input_schema": np.array("unknown")},
                       {"w1": np.zeros((5, 8))},
                       {"b2": np.array([np.nan])},
                       {"b1": np.array([None] * 8, dtype=object)}):
            self.write_archive(**fields)
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                NeuralNetwork.load(self.path)
        np.savez(self.path, w1=np.zeros((8, 5)))
        with self.assertRaisesRegex(ValueError, "fields"):
            NeuralNetwork.load(self.path)

    def test_missing_file_wrong_extension_and_corrupt_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            NeuralNetwork.load(self.path)
        with self.assertRaises(ValueError):
            NeuralNetwork(seed=0).save(self.path.with_suffix(".txt"))
        with self.assertRaises(ValueError):
            NeuralNetwork.load(self.path.with_suffix(".txt"))
        self.path.write_bytes(b"not a numpy archive")
        with self.assertRaises(ValueError):
            NeuralNetwork.load(self.path)

    def test_failed_save_preserves_existing_file_and_cleans_temporary_file(self) -> None:
        NeuralNetwork(seed=0).save(self.path)
        original = self.path.read_bytes()
        with patch("ai.neural_network.np.savez_compressed", side_effect=OSError("disk error")):
            with self.assertRaises(OSError):
                NeuralNetwork(seed=1).save(self.path)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])


if __name__ == "__main__":
    unittest.main()
