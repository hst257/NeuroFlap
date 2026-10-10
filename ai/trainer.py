"""Incremental neuroevolution, champion tracking and exact generation-boundary resume."""

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import tempfile
import zipfile
import numpy as np
from config import DEFAULT_SETTINGS, Settings
from .evaluator import BatchEpisode
from .genetic_algorithm import GeneticAlgorithm, TrainingConfig, from_genome
from .inputs import INPUT_SCHEMA
from .neural_network import NeuralNetwork


class Trainer:
    def __init__(self, config: TrainingConfig = TrainingConfig(), settings: Settings = DEFAULT_SETTINGS,
                 output_dir: str | Path = "models", initial: NeuralNetwork | None = None,
                 population: np.ndarray | None = None) -> None:
        self.config, self.settings = config, settings
        self.output_dir = Path(output_dir)
        self.ga = GeneticAlgorithm(config)
        if population is not None and np.asarray(population).dtype.kind not in "fiu":
            raise ValueError("Population parameters must be real numbers")
        self.population = self.ga.initialize(initial) if population is None else np.array(population, dtype=np.float64, copy=True)
        self._validate_population()
        self.generation = 0
        self.history: list[dict] = []
        self.best_genome: np.ndarray | None = None
        self.best_fitness: float | None = None
        self.best_generation: int | None = None
        self.last_completed_episode: BatchEpisode | None = None
        self._begin_generation()

    def _validate_population(self) -> None:
        if (self.population.shape != (self.config.population_size, 57)
                or not np.isfinite(self.population).all()
                or np.any(np.abs(self.population) > self.config.weight_limit)):
            raise ValueError("Invalid checkpoint population shape, values or weight bounds")

    def _begin_generation(self) -> None:
        self.seed_index = 0
        self.seed_results = []
        self.episode = BatchEpisode(self.population, self.config.evaluation_seeds[0], self.config.max_steps, self.settings)

    def advance(self, tick_budget: int = 1, stop_generation: int | None = None) -> None:
        for _ in range(tick_budget):
            if stop_generation is not None and self.generation >= stop_generation:
                return
            self.episode.step()
            if self.episode.done:
                self.seed_results.append(self.episode.results())
                self.seed_index += 1
                if self.seed_index == len(self.config.evaluation_seeds):
                    self.last_completed_episode = self.episode
                    self._complete_generation()
                    self._begin_generation()
                else:
                    self.episode = BatchEpisode(self.population, self.config.evaluation_seeds[self.seed_index],
                                                self.config.max_steps, self.settings)

    def _complete_generation(self) -> None:
        fitness = np.mean([[result.fitness(self.config, self.settings) for result in seed]
                           for seed in self.seed_results], axis=0)
        scores = np.array([[result.score for result in seed] for seed in self.seed_results])
        flaps = np.array([[result.flaps for result in seed] for seed in self.seed_results])
        winner = int(np.argmax(fitness))
        self.generation += 1
        improved = self.best_fitness is None or float(fitness[winner]) > self.best_fitness
        if improved:
            self.best_fitness = float(fitness[winner])
            self.best_genome = self.population[winner].copy()
            self.best_generation = self.generation
        self.history.append({
            "generation": self.generation, "best_fitness": float(fitness[winner]),
            "average_fitness": float(np.mean(fitness)), "highest_score": int(scores.max()),
            "champion_average_score": float(scores[:, winner].mean()),
            "champion_average_flaps": float(flaps[:, winner].mean()),
            "all_time_best_fitness": self.best_fitness,
        })
        self.population = self.ga.breed(self.population, fitness)
        # Population and RNG now describe the NEXT generation: resume is exact.
        self.save_checkpoint()
        if improved:
            self.save_champion()
        self._write_history()

    def champion(self) -> NeuralNetwork:
        if self.best_genome is None:
            raise ValueError("No generation has been evaluated yet")
        return from_genome(self.best_genome)

    def save_champion(self) -> None:
        self.champion().save(self.output_dir / "best.npz")

    def _write_history(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "history.json").write_text(json.dumps(self.history, indent=2, allow_nan=False), encoding="utf-8")

    def save_checkpoint(self, path: str | Path | None = None) -> None:
        destination = Path(path) if path is not None else self.output_dir / "checkpoint.npz"
        if destination.suffix.lower() != ".npz":
            raise ValueError("Checkpoint path must end in .npz")
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format_version": 1, "input_schema": INPUT_SCHEMA,
            "config": self.config.to_dict(), "settings": asdict(self.settings),
            "generation": self.generation, "history": self.history,
            "rng_state": self.ga.rng.bit_generator.state,
            "best_fitness": self.best_fitness, "best_generation": self.best_generation,
        }
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".npz", delete=False) as stream:
                temp_path = Path(stream.name)
                np.savez_compressed(stream, population=self.population,
                                    champion=np.array([]) if self.best_genome is None else self.best_genome,
                                    metadata=np.array(json.dumps(metadata, allow_nan=False)))
            os.replace(temp_path, destination)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    @classmethod
    def load_checkpoint(cls, path: str | Path, output_dir: str | Path | None = None) -> "Trainer":
        source = Path(path)
        try:
            with np.load(source, allow_pickle=False) as archive:
                if set(archive.files) != {"metadata", "population", "champion"}:
                    raise ValueError("Checkpoint fields are missing or unrecognized")
                raw = archive["metadata"]
                if raw.shape != () or raw.dtype.kind != "U":
                    raise ValueError("Checkpoint metadata must be a text scalar")
                meta = json.loads(raw.item())
                if meta["format_version"] != 1 or meta["input_schema"] != INPUT_SCHEMA:
                    raise ValueError("Incompatible checkpoint format or input schema")
                trainer = cls(TrainingConfig(**meta["config"]), Settings(**meta["settings"]),
                              source.parent if output_dir is None else output_dir, population=archive["population"])
                generation = meta["generation"]
                if type(generation) is not int or generation < 0 or len(meta["history"]) != generation:
                    raise ValueError("Checkpoint generation and history disagree")
                for index, row in enumerate(meta["history"], 1):
                    if row["generation"] != index or any(not math.isfinite(row[key]) for key in (
                        "best_fitness", "average_fitness", "highest_score", "all_time_best_fitness",
                    )):
                        raise ValueError("Invalid generation history")
                champion = archive["champion"]
                if generation == 0:
                    if champion.shape != (0,) or meta["best_fitness"] is not None or meta["best_generation"] is not None:
                        raise ValueError("Unevaluated checkpoint contains a champion")
                else:
                    from_genome(champion)
                    if (not math.isfinite(meta["best_fitness"])
                            or type(meta["best_generation"]) is not int
                            or not 1 <= meta["best_generation"] <= generation
                            or np.any(np.abs(champion) > trainer.config.weight_limit)):
                        raise ValueError("Invalid checkpoint champion")
                    trainer.best_genome = champion.copy()
                trainer.generation, trainer.history = generation, meta["history"]
                trainer.best_fitness, trainer.best_generation = meta["best_fitness"], meta["best_generation"]
                trainer.ga.rng.bit_generator.state = meta["rng_state"]
                trainer._begin_generation()
                return trainer
        except (ValueError, TypeError, KeyError, EOFError, AttributeError, zipfile.BadZipFile) as exc:
            raise ValueError(f"Invalid training checkpoint {source}: {exc}") from exc

    def run(self, generations: int, report=print) -> None:
        if generations < 1:
            raise ValueError("generations must be positive")
        target = self.generation + generations
        try:
            while self.generation < target:
                previous = self.generation
                self.advance(128, stop_generation=target)
                if self.generation != previous:
                    # A budget can finish more than one short generation.
                    for row in self.history[previous:self.generation]:
                        report(f"GEN {row['generation']:03d}  best={row['best_fitness']:.2f}  "
                               f"avg={row['average_fitness']:.2f}  pipes={row['highest_score']}  "
                               f"champion_flaps={row['champion_average_flaps']:.1f}")
        finally:
            self.save_checkpoint()
            if self.best_genome is not None:
                self.save_champion()
