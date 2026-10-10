"""A manual 5 → 8 → 1 feedforward network, with no training framework."""

from collections.abc import Mapping
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np
from numpy.typing import ArrayLike, NDArray
from .inputs import INPUT_SCHEMA

PARAMETER_SHAPES = {"w1": (8, 5), "b1": (8,), "w2": (1, 8), "b2": (1,)}
FORMAT_VERSION = 1


def sigmoid(values: ArrayLike) -> NDArray[np.float64]:
    """Stable sigmoid: exp only sees nonpositive values, avoiding overflow."""
    x = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError("Sigmoid inputs must be finite")
    exp = np.exp(-np.abs(x))
    return np.where(x >= 0, 1.0 / (1.0 + exp), exp / (1.0 + exp))


class NeuralNetwork:
    input_size = 5
    hidden_size = 8
    output_size = 1
    parameter_count = 57

    def __init__(self, seed: int | None = None,
                 parameters: Mapping[str, ArrayLike] | None = None) -> None:
        if parameters is None:
            rng = np.random.default_rng(seed)
            # Xavier normal initialization controls variance entering each layer.
            parameters = {
                "w1": rng.normal(0.0, np.sqrt(2.0 / (5 + 8)), (8, 5)),
                "b1": np.zeros(8),
                "w2": rng.normal(0.0, np.sqrt(2.0 / (8 + 1)), (1, 8)),
                "b2": np.zeros(1),
            }
        validated = self._validate_parameters(parameters)
        self.w1 = validated["w1"]
        self.b1 = validated["b1"]
        self.w2 = validated["w2"]
        self.b2 = validated["b2"]

    @staticmethod
    def _validate_parameters(parameters: Mapping[str, ArrayLike]) -> dict[str, NDArray[np.float64]]:
        if set(parameters) != set(PARAMETER_SHAPES):
            raise ValueError("Parameters must contain exactly w1, b1, w2 and b2")
        result = {}
        for name, shape in PARAMETER_SHAPES.items():
            raw = np.asarray(parameters[name])
            if raw.dtype.kind not in "fiu":
                raise ValueError(f"{name} must contain real numeric values")
            value = np.array(raw, dtype=np.float64, copy=True)
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain finite values")
            result[name] = value
        return result

    def to_parameters(self) -> dict[str, NDArray[np.float64]]:
        """Return independent copies, so editing an exported model cannot alias it."""
        return self._validate_parameters({name: getattr(self, name) for name in PARAMETER_SHAPES})

    def forward(self, inputs: ArrayLike) -> float:
        """Compute one output in [0, 1] from a flat, normalized observation."""
        raw = np.asarray(inputs)
        if raw.dtype.kind not in "fiu":
            raise ValueError("Network inputs must be real numeric values")
        x = np.asarray(raw, dtype=np.float64)
        if x.shape != (self.input_size,):
            raise ValueError(f"Expected input shape (5,), got {x.shape}")
        if not np.all(np.isfinite(x)):
            raise ValueError("Network inputs must be finite")
        if np.any(np.abs(x) > 1.0):
            raise ValueError("Network inputs must be normalized to [-1, 1]")
        try:
            with np.errstate(over="raise", invalid="raise"):
                hidden = np.tanh(self.w1 @ x + self.b1)
                output = sigmoid(self.w2 @ hidden + self.b2)
        except FloatingPointError as exc:
            raise ValueError("Network arithmetic overflowed; check its parameters") from exc
        return float(output[0])

    def should_flap(self, inputs: ArrayLike) -> bool:
        # Exactly 0.5 means no flap, as specified in the decision contract.
        return self.forward(inputs) > 0.5

    def save(self, path: str | Path) -> None:
        """Atomically store numeric arrays plus architecture and feature metadata."""
        destination = Path(path)
        if destination.suffix.lower() != ".npz":
            raise ValueError("Model path must end with .npz")
        parameters = self.to_parameters()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".npz", delete=False) as stream:
                temp_path = Path(stream.name)
                np.savez_compressed(
                    stream, **parameters,
                    format_version=np.array(FORMAT_VERSION, dtype=np.int64),
                    architecture=np.array([5, 8, 1], dtype=np.int64),
                    input_schema=np.array(INPUT_SCHEMA),
                )
            os.replace(temp_path, destination)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path) -> "NeuralNetwork":
        """Reject incompatible, malformed or nonfinite models; never load pickle."""
        source = Path(path)
        if source.suffix.lower() != ".npz":
            raise ValueError("Model path must end with .npz")
        try:
            with np.load(source, allow_pickle=False) as archive:
                expected = set(PARAMETER_SHAPES) | {"format_version", "architecture", "input_schema"}
                if set(archive.files) != expected:
                    raise ValueError("Model fields are missing or unrecognized")
                version = archive["format_version"]
                if version.shape != () or version.dtype.kind not in "iu" or version.item() != FORMAT_VERSION:
                    raise ValueError("Unsupported model format version")
                architecture = archive["architecture"]
                if architecture.dtype.kind not in "iu" or not np.array_equal(architecture, [5, 8, 1]):
                    raise ValueError("Model architecture must be 5 -> 8 -> 1")
                schema = archive["input_schema"]
                if schema.shape != () or schema.dtype.kind != "U" or schema.item() != INPUT_SCHEMA:
                    raise ValueError("Model input normalization schema is incompatible")
                return cls(parameters={name: archive[name] for name in PARAMETER_SHAPES})
        except (ValueError, TypeError, KeyError, EOFError, zipfile.BadZipFile) as exc:
            raise ValueError(f"Invalid model {source}: {exc}") from exc
