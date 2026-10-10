"""Selection, uniform crossover and Gaussian mutation of 57 network parameters."""

from dataclasses import asdict, dataclass
import math
import numpy as np
from .neural_network import NeuralNetwork, PARAMETER_SHAPES


@dataclass(frozen=True)
class TrainingConfig:
    population_size: int = 100
    elite_count: int = 5
    tournament_size: int = 5
    immigrant_count: int = 5
    crossover_rate: float = 0.7
    mutation_rate: float = 0.15
    mutation_scale: float = 0.25
    weight_limit: float = 8.0
    max_steps: int = 1800
    pipe_reward: float = 1000.0
    survival_reward: float = 1.0
    progress_reward: float = 2.0
    training_seed: int = 0
    evaluation_seeds: tuple[int, ...] = (42, 7, 19)
    unseen_seeds: tuple[int, ...] = (101, 202, 303, 404, 505)

    def __post_init__(self) -> None:
        for name in ("population_size", "elite_count", "tournament_size", "immigrant_count", "max_steps", "training_seed"):
            value = getattr(self, name)
            if type(value) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.population_size < 2 or not 1 <= self.elite_count < self.population_size:
            raise ValueError("Use at least two individuals and 1 <= elite_count < population_size")
        if not 1 <= self.tournament_size <= self.population_size:
            raise ValueError("tournament_size must fit the population")
        if not 0 <= self.immigrant_count < self.population_size - self.elite_count:
            raise ValueError("Leave at least one offspring slot after elites and immigrants")
        if self.max_steps < 1 or self.training_seed < 0:
            raise ValueError("max_steps must be positive and training_seed nonnegative")
        for name in ("crossover_rate", "mutation_rate", "mutation_scale", "weight_limit",
                     "pipe_reward", "survival_reward", "progress_reward"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.crossover_rate > 1 or self.mutation_rate > 1 or self.weight_limit == 0 or self.pipe_reward == 0:
            raise ValueError("Rates must be <= 1; weight_limit and pipe_reward must be positive")
        for name in ("evaluation_seeds", "unseen_seeds"):
            values = tuple(getattr(self, name))
            if not values or any(type(seed) is not int for seed in values) or len(set(values)) != len(values):
                raise ValueError(f"{name} must contain distinct integer seeds")
            object.__setattr__(self, name, values)
        if set(self.evaluation_seeds) & set(self.unseen_seeds):
            raise ValueError("Unseen seeds must be separate from training evaluation seeds")

    def to_dict(self) -> dict:
        return asdict(self)


def to_genome(network: NeuralNetwork) -> np.ndarray:
    return np.concatenate([value.ravel() for value in network.to_parameters().values()])


def from_genome(genome: np.ndarray) -> NeuralNetwork:
    if np.asarray(genome).dtype.kind not in "fiu":
        raise ValueError("Genome parameters must be real numbers")
    values = np.asarray(genome, dtype=np.float64)
    if values.shape != (57,) or not np.isfinite(values).all():
        raise ValueError("A genome must have 57 finite parameters")
    offset = 0
    parameters = {}
    for name, shape in PARAMETER_SHAPES.items():
        count = math.prod(shape)
        parameters[name] = values[offset:offset + count].reshape(shape)
        offset += count
    return NeuralNetwork(parameters=parameters)


class GeneticAlgorithm:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.training_seed)

    def fresh_genome(self) -> np.ndarray:
        seed = int(self.rng.integers(0, 2**32))
        return np.clip(to_genome(NeuralNetwork(seed=seed)), -self.config.weight_limit, self.config.weight_limit)

    def initialize(self, initial: NeuralNetwork | None = None) -> np.ndarray:
        population = np.stack([self.fresh_genome() for _ in range(self.config.population_size)])
        if initial is not None:
            genome = to_genome(initial)
            if np.any(np.abs(genome) > self.config.weight_limit):
                raise ValueError("Initial model exceeds training weight_limit")
            population[0] = genome
        return population

    def select(self, fitness: np.ndarray) -> int:
        candidates = self.rng.choice(len(fitness), self.config.tournament_size, replace=False)
        return int(candidates[np.argmax(fitness[candidates])])

    def crossover(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self.rng.random() < self.config.crossover_rate:
            return np.where(self.rng.random(a.shape) < 0.5, a, b)
        return a.copy()

    def mutate(self, genome: np.ndarray) -> np.ndarray:
        mask = self.rng.random(genome.shape) < self.config.mutation_rate
        child = genome + mask * self.rng.normal(0, self.config.mutation_scale, genome.shape)
        return np.clip(child, -self.config.weight_limit, self.config.weight_limit)

    def breed(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        c = self.config
        if population.shape != (c.population_size, 57) or not np.isfinite(population).all():
            raise ValueError("Population must contain finite genomes with shape (population_size, 57)")
        if fitness.shape != (c.population_size,) or not np.isfinite(fitness).all():
            raise ValueError("Fitness must contain one finite value per individual")
        order = np.argsort(-fitness, kind="stable")
        children = [genome.copy() for genome in population[order[:c.elite_count]]]
        while len(children) < c.population_size - c.immigrant_count:
            a, b = population[self.select(fitness)], population[self.select(fitness)]
            children.append(self.mutate(self.crossover(a, b)))
        children.extend(self.fresh_genome() for _ in range(c.immigrant_count))
        return np.stack(children)
