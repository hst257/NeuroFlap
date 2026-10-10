"""Batched independent birds on a shared seeded course, using the game physics."""

from dataclasses import dataclass
import math
import numpy as np
from config import DEFAULT_SETTINGS, Settings
from game.environment import Environment
from .genetic_algorithm import TrainingConfig, to_genome
from .neural_network import NeuralNetwork, sigmoid


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    score: int
    ticks: int
    distance: float
    flaps: int
    death_reason: str | None
    step_limit_reached: bool

    def fitness(self, config: TrainingConfig, settings: Settings) -> float:
        return (self.score * config.pipe_reward + self.ticks * settings.dt * config.survival_reward
                + self.distance / settings.pipe_spacing * config.progress_reward)


class BatchEpisode:
    def __init__(self, population: np.ndarray, seed: int, max_steps: int,
                 settings: Settings = DEFAULT_SETTINGS) -> None:
        if np.asarray(population).dtype.kind not in "fiu":
            raise ValueError("Population parameters must be real numbers")
        values = np.asarray(population, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 57 or not len(values) or not np.isfinite(values).all():
            raise ValueError("Expected a nonempty finite population with shape (n, 57)")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.settings, self.seed, self.max_steps = settings, seed, max_steps
        self.world = Environment(settings, seed)
        n = len(values)
        # Retain the exact genomes that produced this episode for view selection.
        # A dashboard never changes these arrays or controls bird decisions.
        self.genomes = values.copy()
        self.genomes.flags.writeable = False
        self.w1 = self.genomes[:, :40].reshape(n, 8, 5)
        self.b1 = self.genomes[:, 40:48]
        self.w2 = self.genomes[:, 48:56]
        self.b2 = self.genomes[:, 56]
        self.y = np.full(n, self.world.bird.y)
        self.velocity = np.zeros(n)
        self.alive = np.ones(n, dtype=bool)
        self.scores = np.zeros(n, dtype=np.int64)
        self.ticks = np.zeros(n, dtype=np.int64)
        self.distances = np.zeros(n)
        self.flaps = np.zeros(n, dtype=np.int64)
        self.outputs = np.full(n, 0.5)
        self.death_reasons = np.full(n, "", dtype="U7")
        self.steps = 0

    @property
    def done(self) -> bool:
        return not self.alive.any() or self.steps >= self.max_steps

    def step(self) -> None:
        if self.done:
            return
        s = self.settings
        pipe = self.world.next_pipe
        inputs = np.column_stack((2 * self.y / s.play_height - 1,
                                  self.velocity / max(abs(s.flap_velocity), s.terminal_velocity),
                                  np.full(len(self.y), (pipe.x - s.bird_x) / s.width),
                                  (pipe.gap_top - self.y) / s.play_height,
                                  (pipe.gap_bottom - self.y) / s.play_height))
        np.clip(inputs, -1, 1, out=inputs)
        hidden = np.tanh(np.einsum("nij,nj->ni", self.w1, inputs) + self.b1)
        outputs = sigmoid(np.einsum("ni,ni->n", self.w2, hidden) + self.b2)
        self.outputs[self.alive] = outputs[self.alive]
        actions = (self.outputs > 0.5) & self.alive
        self.velocity[actions] = s.flap_velocity
        self.flaps[actions] += 1
        self.ticks[self.alive] += 1
        self.steps += 1
        # This common conservative bound is also used in Environment.step.
        speed_bound = max(abs(s.flap_velocity), s.terminal_velocity) + s.gravity * s.dt
        substeps = max(1, math.ceil((speed_bound + s.pipe_speed) * s.dt / (s.bird_radius * 0.5)))
        dt = s.dt / substeps
        for _ in range(substeps):
            self.velocity[self.alive] = np.minimum(self.velocity[self.alive] + s.gravity * dt, s.terminal_velocity)
            self.y[self.alive] += self.velocity[self.alive] * dt
            self.distances[self.alive] += s.pipe_speed * dt
            for pair in self.world.pipes:
                pair.update(dt, s)
            self.world.distance += s.pipe_speed * dt
            self.world._fill_pipes()
            self._kill((self.y - s.bird_radius <= 0), "ceiling")
            self._kill((self.y + s.bird_radius >= s.play_height), "ground")
            for pair in self.world.pipes:
                dx = s.bird_x - max(pair.x, min(s.bird_x, pair.right))
                if abs(dx) <= s.bird_radius:
                    for top, bottom in ((0, pair.gap_top), (pair.gap_bottom, s.play_height)):
                        dy = self.y - np.clip(self.y, top, bottom)
                        self._kill(dx * dx + dy * dy <= s.bird_radius**2, "pipe")
            for pair in self.world.pipes:
                if not pair.passed and pair.right < s.bird_x - s.bird_radius:
                    self.scores[self.alive] += 1
                    pair.passed = True
            self.world.pipes = [pair for pair in self.world.pipes if pair.right >= 0]
            if not self.alive.any():
                break

    def _kill(self, collision: np.ndarray, reason: str) -> None:
        deaths = collision & self.alive
        self.death_reasons[deaths] = reason
        self.alive[deaths] = False

    def results(self) -> list[EpisodeResult]:
        if not self.done:
            raise ValueError("Episode is still running")
        return [EpisodeResult(self.seed, int(self.scores[i]), int(self.ticks[i]),
                              float(self.distances[i]), int(self.flaps[i]),
                              str(self.death_reasons[i]) or None, bool(self.alive[i]))
                for i in range(len(self.y))]


def evaluate_network(network: NeuralNetwork, seeds: tuple[int, ...], max_steps: int = 1800,
                     settings: Settings = DEFAULT_SETTINGS) -> list[EpisodeResult]:
    results = []
    for seed in seeds:
        episode = BatchEpisode(to_genome(network)[None, :], seed, max_steps, settings)
        while not episode.done:
            episode.step()
        results.append(episode.results()[0])
    return results


def evaluation_summary(results: list[EpisodeResult]) -> dict:
    if not results:
        raise ValueError("At least one evaluation episode is required")
    scores = [result.score for result in results]
    return {"seeds": [result.seed for result in results], "scores": scores,
            "average_score": float(np.mean(scores)), "median_score": float(np.median(scores)),
            "maximum_score": max(scores), "zero_score_failure_rate": sum(score == 0 for score in scores) / len(scores),
            "death_rate": sum(result.death_reason is not None for result in results) / len(results),
            "step_limit_rate": sum(result.step_limit_reached for result in results) / len(results)}
