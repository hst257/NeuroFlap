# Phase 2 — the bird's decision network

The network has **5 inputs → 8 hidden neurons → 1 output**, and **57 adjustable
parameters**. It uses NumPy for arrays and arithmetic, without TensorFlow,
PyTorch, NEAT, gradient descent or backpropagation. This module performs inference;
the [Phase 3 trainer](training.md) now evolves its parameters.

## 1. Turn game state into five numbers

`ai/inputs.py` returns a flat `float64` vector with shape `(5,)`. Let `H` be the
playable height above the ground, `W` the window width, and
`V = max(abs(flap_velocity), terminal_velocity)`.

| Index | Input | Normalization before clipping |
| --- | --- | --- |
| 0 | Bird vertical position | `2 * bird.y / H - 1` |
| 1 | Bird vertical velocity | `bird.velocity / V` |
| 2 | Horizontal distance to next pipe's leading edge | `(pipe.x - bird.x) / W` |
| 3 | Distance to upper gap edge | `(pipe.gap_top - bird.y) / H` |
| 4 | Distance to lower gap edge | `(pipe.gap_bottom - bird.y) / H` |

Every input is clipped to `[-1, 1]`. The bird center is the reference point.
Screen y increases downward: upward velocity is negative, and a negative gap
delta means that edge is above the bird. Vertical position is `-1` at the top,
`0` halfway down and `+1` at the ground line.

For example, at `y = 330`, velocity `340`, pipe x `300`, gap top `220`, and gap
bottom `402`, with default settings, the observation is approximately:

```text
[0.0, 0.5, 0.35833, -0.16667, 0.10909]
```

The current pipe stays the target until its right edge has cleared the entire
bird. Its horizontal distance can therefore become negative while the bird is
still inside the gap. The next pair then becomes the target. This agrees with
collision and scoring geometry.

Normalization keeps a 600-pixel position from overwhelming a small velocity or
gap signal. Centered inputs also suit tanh's signed output. These fixed scales
depend only on game settings; they need no statistics from future episodes.
Clipping handles distant pipes and bounds values even when game settings change.
NaN or infinite observations are rejected rather than silently converted.

The controller receives bird motion and the next pipe's geometry. It does not
use the course seed, future random draws, score, or collision outcomes to choose
an action. Reading the observation does not modify the simulation.

## 2. What weights and biases mean

Each hidden neuron has five weights, one per input. A weight controls how much
an input contributes to that neuron's activation; a negative weight reverses
the contribution. A bias adds a constant offset, moving the activation threshold.

For hidden neuron `j`:

```text
z[j] = w1[j,0]*x[0] + ... + w1[j,4]*x[4] + b1[j]
h[j] = tanh(z[j])
```

`tanh` maps this sum into `[-1, 1]`. Its nonlinearity allows hidden neurons to
respond to combinations such as downward motion near the lower gap edge.
Randomly initialized neurons do not automatically have useful interpretations.

## 3. Forward propagation and matrix shapes

NumPy's `@` operator performs the weighted sums together:

```text
x                         shape (5,)
w1                        shape (8, 5)
b1                        shape (8,)
h = tanh(w1 @ x + b1)      shape (8,)

w2                        shape (1, 8)
b2                        shape (1,)
z2 = w2 @ h + b2          shape (1,)
output = sigmoid(z2)       shape (1,) -> Python float
```

The inner dimensions must match: `(8, 5) @ (5,)` produces eight sums, and
`(1, 8) @ (8,)` combines those eight hidden activations into one sum. Bias vectors
add one offset per neuron. The parameter count is:

```text
8*5 hidden weights + 8 hidden biases + 1*8 output weights + 1 output bias = 57
```

For a small worked example, suppose only the first hidden neuron contributes,
its first weight is `2`, its bias is `-0.5`, and `x[0] = 0.5`:

```text
z[0] = 2*0.5 - 0.5 = 0.5
h[0] = tanh(0.5) ≈ 0.46212
z2 = 3*h[0] + 0.1 ≈ 1.48635
output = sigmoid(z2) ≈ 0.81553
```

## 4. Decide whether to flap

Sigmoid is `1 / (1 + exp(-z2))`; it maps the output sum into `[0, 1]`. The code
uses an equivalent expression with `exp(-abs(z2))` to avoid overflowing when
`z2` is very negative. This output is an action signal, not a calibrated estimate
of success probability.

**Output greater than 0.5 means flap.** Exactly `0.5` means no flap. A zeroed
network always returns `0.5`, so it never flaps. In network mode the controller
reads the current state and runs this forward pass once before every fixed tick,
including each tick used to catch up after a slow rendered frame. Human inputs
cannot insert extra flaps during a network flight.

Watch mode displays the latest output and decision, flap count and attempt.
After a loss it shows a 1.5-second countdown and starts another attempt. Esc
cancels that countdown; `--no-auto-restart` keeps the result screen open. These
restarts do not change the network parameters or train the bird. With the same
course seed, the same network repeats the same actions. Headless mode still runs
one episode and reports its output range and flap count.

The bundled seed-0 model produces only outputs below 0.5 on course seed 42
(approximately 0.302 to 0.426), so it never flaps and hits the ground after 46
ticks. The network is evaluated on every tick. Training is the step that will
search for better weights and behavior.

## 5. Initialization and persistence

Weights start from a normal distribution with standard deviation
`sqrt(2 / (fan_in + fan_out))`, often called Xavier initialization. It limits the
starting variance so hidden neurons are less likely to begin saturated. All
biases start at zero. A local NumPy generator makes initial weights repeatable
for a given network seed without changing the game's pipe generator.

```python
from ai.neural_network import NeuralNetwork
from ai.inputs import encode_inputs
from game.environment import Environment

network = NeuralNetwork(seed=0)
env = Environment(seed=42)
inputs = encode_inputs(env)
output = network.forward(inputs)
env.step(network.should_flap(inputs))
network.save("models/example.npz")
restored = NeuralNetwork.load("models/example.npz")
assert restored.forward(inputs) == output
```

`.npz` is a compressed NumPy archive. It stores `w1`, `b1`, `w2`, `b2`, a format
version, `[5, 8, 1]` architecture, and input-schema identifier
`neuroflap.inputs.v1`. The schema fixes input order, signs, scaling and clipping.
Changing that contract requires a new identifier. The loader checks metadata,
shapes and finite real values, and uses `allow_pickle=False`. Saving writes a
temporary file beside the destination, then replaces the destination after the
archive is complete. A failed write leaves an existing model intact.

Model files store parameters and their interpretation, not training progress.
Generation statistics, fitness histories and resumable training checkpoints
belong to Phase 3. `to_parameters()` returns independent copies, and the
constructor accepts a validated parameter dictionary for experiments.

## Run and test

From the project folder in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py --create-model models/untrained.npz --network-seed 0
.\.venv\Scripts\python.exe main.py --model models/untrained.npz --seed 42
.\.venv\Scripts\python.exe main.py --model models/untrained.npz --headless --seed 42 --steps 600
.\.venv\Scripts\python.exe -m unittest discover -v
```

The tests check matrix shapes, independent scalar calculations, exact decision
thresholds, sigmoid stability, normalization and signs, seeded replay, save/load
equivalence, malformed files and failed saves. The existing physics and menu
checks also run, including integration of network decisions into the game loop.

Before Phase 3, be comfortable tracing an observation through the two matrix
multiplications, identifying what a weight and bias change, and explaining why
a random network can crash even though its mathematics is correct. Genetic
evolution changes these 57 parameters based on performance; the forward
pass itself will stay the same.
