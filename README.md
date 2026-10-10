# NeuroFlap

A playable, original Flappy Bird-inspired game in Python and Pygame, with a
manually implemented NumPy neural network and a genetic algorithm trainer.
**Phases 1–3 are implemented**, with an interactive population view to observe
learning. All artwork is drawn in code; no downloaded assets are needed.

[View the four-screen visual preview](docs/preview.png).
[View the network-control preview](docs/network-preview.png).
[View learning and the trained champion](docs/training-preview.png).

## Start actual learning

**Restarting a fixed model does not train it.** Use the training mode to evolve
the weights, or press **T** in the normal game / network watch mode:

```powershell
cd Z:\Code\NeuroFlap
.\.venv\Scripts\python.exe main.py --train --generations 50
```

This opens a live population view and evolves 100 networks. The generation
counter advances after every bird has been evaluated on all training courses.
Space pauses, +/- changes speed, B shows the current leader, R toggles scene
rendering, S saves, W stops training and watches the available champion, and Esc
saves and exits. The interactive view has a frame budget to keep controls
responsive; its displayed speed is an upper bound depending on hardware.

Train faster without opening a window:

```powershell
.\.venv\Scripts\python.exe main.py --train --headless --generations 50
```

Watch the trained champion:

```powershell
.\.venv\Scripts\python.exe main.py --model models/best.npz --seed 42
```

Continue the saved population for 50 additional generations:

```powershell
.\.venv\Scripts\python.exe main.py --train --resume models/checkpoint.npz --generations 50
```

`best.npz` is the playable model. `checkpoint.npz` contains the population, RNG
state, configuration, seeds, fitness, generation and history. `history.json`
records generation statistics; `evaluation.json` records held-out results.
Models and checkpoints use different file formats: load a model with `--model`
and a population checkpoint with `--resume`. The original `untrained.npz` remains
an untrained example.

A fresh `--train` run replaces the training outputs in `models/`; use `--resume`
to retain the population or `--output-dir another-run` to keep separate runs.
Pressing T from a fixed model starts a fresh population that includes that model.
Pressing T after using W to watch a live training champion resumes its checkpoint.

The bundled 50-generation run's champion scored 22 with 168 flaps on seed 42
over 1,800 ticks. On held-out seeds 101, 202, 303, 404 and 505 it scored
`[22, 22, 1, 22, 22]` (average 17.8). Four runs reached the 30-second step limit;
one died after one pipe. These bounded results demonstrate learning, not mastery.
Full details are in the [Phase 3 walkthrough](docs/training.md).

## Run on Windows

Requires Python 3.11 or newer. In PowerShell:

```powershell
cd Z:\Code\NeuroFlap
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

Using the venv's Python directly avoids PowerShell activation-policy issues.
For a reproducible pipe course:

```powershell
.\.venv\Scripts\python.exe main.py --seed 42
```

On other platforms use `python3 -m venv .venv` and `.venv/bin/python` instead.
If a display cannot be opened, the launcher reports the Pygame error. The
simulation itself needs only Python's standard library.

## Controls

| Action | Control |
| --- | --- |
| Select Play / Quit or Retry / Menu | Click, or arrows / Tab then Enter |
| Open a flight from the menu | Space or Play |
| Start flying / flap | Space, left click, or Enter |
| Restart | R; Space also restarts and flaps after game over |
| Return to the menu | Esc |
| Quit from the menu | Quit button or Esc |

Play opens a ready screen. The bird waits until the first flap. Hit the ceiling,
ground or a pipe and the flight ends. Clear a pipe with the whole bird to earn
one point. The best score lasts for the current app session. Hold-to-flap is
disabled: each key press or click creates one flap.

## Create and try a neural network

Install the updated requirements in the same virtual environment, then create an
**untrained** network. `--network-seed` controls the initial weights independently
of `--seed`, which controls the pipe course:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py --create-model models/untrained.npz --network-seed 0
.\.venv\Scripts\python.exe main.py --model models/untrained.npz --seed 42
```

Select Play to watch that network. It starts immediately and chooses whether to
flap once per simulation tick. Space, click and Enter do not override the
network during flight. Restart and menu controls still work. The screen labels
this mode as NETWORK PILOT and displays the current output, decision, flap count
and attempt number. After a loss it restarts automatically after 1.5 seconds.
Esc cancels the pending restart and returns to the menu. Human mode keeps manual
restarts. Add `--no-auto-restart` to keep network mode on the game-over screen;
the delay is configurable as `network_restart_delay` in `config.py`.

Random weights have not learned anything, so a quick crash is expected. Phase 2
implements decisions and persistence; it does not improve the weights.
The bundled seed-0 model chooses zero flaps on course seed 42: its outputs range
from about 0.302 to 0.426, below the required `> 0.5` threshold. That is a valid
decision from random weights, not a failure to run inference. Automatic restarts
reuse the same weights. With a fixed course seed they repeat the same behavior;
learning requires the training mode above. Watch mode labels its weights as fixed
and offers T to start training.

For inference without opening a window:

```powershell
.\.venv\Scripts\python.exe main.py --model models/untrained.npz --headless --seed 42 --steps 600
```

This prints the score, tick count, flap count, network output range, cause of
death and whether the step limit was reached. Headless mode runs one bounded
episode; automatic restarts apply to the visible watch mode. Headless inference
imports NumPy and the simulation, without Pygame.
Malformed or incompatible model files produce a readable error and exit code 1.
Creating a model exits after saving, without opening a game window.

Read the [Phase 2 walkthrough](docs/neural_network.md) for input normalization,
matrix shapes, a worked example and a direct Python API example.

## Structure and design choices

```text
main.py                  CLI and friendly launch errors
config.py                Validated, immutable game settings
game/
  application.py         Menu / ready / playing / game-over state and clock
  environment.py         Seeded simulation, collision checks, scoring
  bird.py                Bird physics and circular collision geometry
  pipe.py                Pipe pairs and rectangular collision geometry
  renderer.py            Original artwork, animation and interface
ai/
  inputs.py              Versioned five-feature normalization
  neural_network.py      Manual forward propagation and .npz persistence
  controller.py          Game state -> inputs -> flap decision
  genetic_algorithm.py   Population, selection, crossover and mutation
  evaluator.py           Batched physics and held-out evaluations
  trainer.py             Generation loop, champion and checkpoint resume
visualization/
  training_dashboard.py  Live learning view and training controls
models/                  Saved .npz models (ignored by Git)
tests/                   Physics, numerical network, persistence and UI checks
requirements.txt         Pygame and NumPy
```

The `game/` package replaces a top-level `game.py`, following the requested final
project structure. `application.py` owns the interactive loop; `environment.py`
owns the simulation. The AI modules import no Pygame code, so training can run
headlessly at accelerated speed. The dashboard alone imports Pygame.

Settings use pixels, pixels/second and pixels/second². Edit the defaults in
`config.py` to change gravity, flap strength, terminal speed, pipe speed, gap or
spacing. Invalid sizes and impossible gaps fail early. Defaults are a 480 × 720
window, 60 FPS, a 182-pixel pipe gap and 190 pixels/second of pipe movement.

The renderer uses a dusk sky, pixel clouds, mint pipes and a coral bird. Visual
cloud drift and wing motion never change simulation state. The bird collision
shape is a circle; decorative sprite rotation does not change that shape. Pipe
caps stay within each pipe's collision width. The scene freezes on death so the
result is easy to read. There is no audio dependency.

## Physics to understand

Screen coordinates increase downward, so gravity is positive and a flap sets
vertical velocity to a negative value. During each small physics step of length
`dt` seconds:

```text
velocity = min(velocity + gravity * dt, terminal_velocity)
y = y + velocity * dt
pipe_x = pipe_x - pipe_speed * dt
```

This is **semi-implicit Euler integration**: update velocity first, then position.
A flap replaces the velocity rather than adding to it, giving a predictable lift.
At 60 Hz the environment advances one 1/60-second tick per `step()` call. Each
tick is split into smaller steps so the combined bird/pipe motion is at most
half the bird radius between collision checks. This reduces tunneling through
obstacles at fast configured speeds. To accelerate future headless simulations,
call `step()` more often; never increase its time step.

Collision uses the closest point of each pipe rectangle to the bird center.
If the squared distance to that point is no greater than `radius²`, they touch.
This avoids the false corner collisions produced by using a square bird box.
Ceiling and ground contact are fatal too. Scoring happens after collision checks
and only once per pipe, after its right edge passes the bird's trailing edge.

The app accumulates wall-clock time and advances fixed ticks. A stalled frame is
limited to 0.1 seconds of catch-up so dragging or suspending the window does not
produce a huge simulation jump. A queued input survives until the next tick.
Rendering targets 60 FPS, subject to hardware and operating-system scheduling.

Each environment owns its own `random.Random(seed)` generator. The same seed,
settings and action sequence reproduce the same simulation. Restart with a seed
replays that course; without a seed it generates a fresh course. Visual effects
use separate randomness. Death makes subsequent `step()` calls harmless.

## Test and inspect

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
.\.venv\Scripts\python.exe main.py --headless --seed 42
```

The offscreen UI tests use SDL's dummy display driver; they do not open windows.
They exercise keyboard and mouse input, network control, window shutdown and all
four application states.
Physics checks cover integration, flaps, terminal speed, circle corners, all
death boundaries, one-time scoring, replay/reset, valid gap generation and fast
collision checks. To run the physics checks without installing Pygame:

```powershell
python -m unittest tests.test_physics -v
python main.py --headless --seed 42 --steps 600
```

Network checks compare NumPy's forward pass against independent scalar sums,
exercise the exact 0.5 decision boundary and extreme sigmoid inputs, and verify
parameter shapes, normalization, deterministic initialization and exact save/load
round trips. They also reject corrupt files, wrong schemas, wrong shapes and
nonfinite parameters, and check that an interrupted save preserves the old file.
To run only the NumPy/physics tests without Pygame:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_physics tests.test_inputs tests.test_neural_network -v
```

Without `--model`, the headless command applies no flaps and stops at death or
the tick limit. With `--model`, it uses network decisions. No training occurs in
either case; `--train` or T selects learning. Training tests verify mutation,
selection and elitism, actual weight changes, scalar/batched physics parity,
exact checkpoint resume, CLI training/evaluation and dashboard controls.
