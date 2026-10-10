# Phase 3 — learning by genetic evolution

The previous 50 retries ran the same model on the same course. Its final output
was consistently about 0.302 because neither its weights nor the course changed.
The simulation performed inference correctly, but no learning algorithm existed.
The project now includes an explicit training mode that changes network weights.

## How a generation works

1. Initialize 100 different networks (57 parameters each).
2. Run every network on the same fixed evaluation seeds: 42, 7 and 19.
3. Average each network's fitness across those courses.
4. Keep the best five unchanged as elites.
5. Select parents using tournaments of five randomly sampled individuals.
6. Make offspring through uniform crossover and Gaussian mutation.
7. Add five fresh random networks to maintain diversity.
8. Evaluate the next population and repeat.

The birds share a pipe course, not their brains. All decisions still use the
five Phase 2 inputs and the exact output threshold `> 0.5`. No random flaps or
handwritten successful policy are inserted into training.

## Fitness

For each course, defaults use:

```text
fitness = 1000 * passed_pipes
        + 1 * survival_seconds
        + 2 * (horizontal_distance / pipe_spacing)
```

Passing a pipe is the main reward. Survival and fractional forward progress give
early generations useful feedback before they clear their first pipe. Because
pipes always move, a bird cannot park indefinitely without facing obstacles.
Survival and distance are correlated here; the separate terms make their roles
explicit and configurable. There is no hidden bonus for choosing a flap.

Each episode stops after death or 1,800 fixed ticks (30 simulated seconds at
60 Hz). Under the default cap, surviving without passing a pipe earns less than
one pipe reward. If you change the cap or rewards, check their relative sizes.
Reaching the cap is a bounded success, not evidence of endless survival.

The same evaluation seeds are reused so scores remain comparable between
generations. Unseen seeds 101, 202, 303, 404 and 505 are used only to report the
champion's generalization; their results never influence parent selection.

## Selection, crossover and mutation

A tournament samples five population indices without replacement and picks the
one with the highest fitness. Repeating tournaments favors better parents while
allowing weaker individuals to contribute occasionally.

With probability 0.7, uniform crossover chooses each parameter from either
parent with equal probability. Otherwise it copies the first parent. Each
offspring parameter is mutated independently with probability 0.15:

```text
child_parameter += Normal(mean=0, standard_deviation=0.25)
```

Values are limited to `[-8, 8]` to avoid uncontrolled parameter growth. Elites
are copied without mutation. The champion is preserved separately when its mean
fitness improves. There is no backpropagation or gradient descent.

A local seeded NumPy generator controls initial population seeds, tournament
sampling, crossover, mutation and immigrants. Initial networks still use
Xavier initialization. Supplying `--model some.npz --train` includes that network
as the first individual alongside fresh random networks.

## Accelerated simulation

`ai/evaluator.py` computes a batch of forward passes with NumPy and advances
independent bird positions and velocities against one seeded pipe course. Each
bird has its own alive flag, score, flap count, time and distance. Dead birds
stop earning rewards. The shared course remains deterministic.

Both the playable environment and batched evaluator use the same fixed tick,
substep bound, gravity, terminal speed, flap velocity, circular collision rule,
pipe movement and scoring rule. The substep bound uses the configured maximum
flap/terminal speed, so a bird's integration does not depend on other population
members. Tests compare batched birds to independent playable environments at
every tick, including a stronger-than-default flap setting.

Headless training has no frame-rate sleep and imports no Pygame. The interactive
dashboard advances multiple ticks per rendered frame and limits time spent in
simulation between UI updates. +/- controls its target acceleration. R skips
the population scene while keeping progress and controls visible.

## Run, resume and watch

```powershell
# Visible learning, 100 birds, 50 generations
.\.venv\Scripts\python.exe main.py --train --generations 50

# Fast training; prints generation statistics
.\.venv\Scripts\python.exe main.py --train --headless --generations 50

# Resume for 50 ADDITIONAL generations
.\.venv\Scripts\python.exe main.py --train --resume models/checkpoint.npz --generations 50

# Watch a model that has learned
.\.venv\Scripts\python.exe main.py --model models/best.npz --seed 42

# Evaluate it on held-out courses
.\.venv\Scripts\python.exe main.py --model models/best.npz --evaluate

# Choose other evaluation seeds or a longer evaluation cap
.\.venv\Scripts\python.exe main.py --model models/best.npz --evaluate --evaluation-seeds 601 602 603 --max-steps 3600
```

Visible controls: Space pauses, +/- changes speed, B shows only the current
leader, R toggles scene rendering, S saves, W stops learning and watches the
champion, and Esc saves and exits. From ordinary play or fixed model watching,
T starts the trainer. Watching a model does not update its parameters.

The best model is `models/best.npz`; keep using `untrained.npz` only when you
want to inspect random initial behavior. A fresh training command overwrites
training outputs in the chosen directory. Use `--output-dir` for separate runs.

## Configuration

Every genetic algorithm and fitness setting is in the validated `TrainingConfig`
dataclass in `ai/genetic_algorithm.py`. Supply any subset as a JSON object through
`--training-config path.json`; unspecified values use the defaults. For example:

```json
{
  "population_size": 100,
  "elite_count": 5,
  "tournament_size": 5,
  "immigrant_count": 5,
  "mutation_rate": 0.15,
  "mutation_scale": 0.25,
  "crossover_rate": 0.7,
  "weight_limit": 8.0,
  "max_steps": 1800,
  "pipe_reward": 1000.0,
  "survival_reward": 1.0,
  "progress_reward": 2.0,
  "training_seed": 0,
  "evaluation_seeds": [42, 7, 19],
  "unseen_seeds": [101, 202, 303, 404, 505]
}
```

CLI shortcuts include `--population`, `--training-seed` and `--max-steps`.
`--population` sets 5% elites, 5% immigrants and tournaments up to size five.
Game physics settings come from `config.py`; checkpoints preserve their exact
values. Resume rejects configuration overrides to avoid changing the experiment
silently. Set `--generations` to the number of additional generations.

## Saving and exact resume

After each completed generation, save:

- `checkpoint.npz`: next population, champion, fitness, generation, configuration,
  game settings, random seeds, RNG state and all generation statistics.
- `best.npz`: the champion's ordinary Phase 2 model format, ready for `--model`.
- `history.json`: readable best/mean fitness, high scores and champion flap counts.
- `evaluation.json`: held-out results after stopping or completing a training run.

Checkpoint arrays and JSON text load with `allow_pickle=False`. A temporary file
is replaced atomically after a completed write. A failed checkpoint write leaves
the previous checkpoint intact. The loader validates versions, feature schema,
population shape, weight bounds, champion, history and RNG state.

Resume starts at the next unevaluated generation. Saving during a partial
generation discards only that generation's incomplete evaluation; it replays
from the start using the saved population. No generation is incorrectly counted
as finished. Tests compare uninterrupted training with save/resume and require
identical populations, RNG states, champions and histories.

## Verified sample run and limits

The saved sample used 100 birds and 50 generations with default settings and
training seed 0. Mean population fitness rose from about 36.5 in generation 1 to
9,580 in generation 50. The champion was first found in generation 4; later
generations retained it while evolving the rest of the population.

On course seed 42, the playable environment recorded 168 flaps and 22 pipes over
1,800 ticks, surviving to the cap. Its output varied across the 0.5 threshold.
The original untrained seed-0 model recorded zero flaps and zero pipes.

On the five unseen courses, scores were `[22, 22, 1, 22, 22]`: average 17.8,
median 22, maximum 22. Four survived to the cap; one died. Zero-score failure rate
was 0%, but death rate was 20%. These are different metrics, and the report
labels them separately. This sample has learned useful behavior, with remaining
generalization failures. It has not demonstrated mastery or unlimited survival.

Run all checks with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
```

Before further work, understand the difference between inference and training,
why a generation evaluates many different networks, why elites stay unchanged,
how mutation explores new weights, and why held-out courses must not be used
to choose parents. Optional plots and deeper analytics can build on the saved
history and evaluation files later.
