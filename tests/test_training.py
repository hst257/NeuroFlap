"""Physics parity, actual weight evolution and exact checkpoint resume."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from ai.controller import NetworkController
from ai.evaluator import BatchEpisode, EpisodeResult, evaluate_network, evaluation_summary
from ai.genetic_algorithm import GeneticAlgorithm, TrainingConfig, from_genome, to_genome
from ai.neural_network import NeuralNetwork
from ai.trainer import Trainer
from config import DEFAULT_SETTINGS
from game.environment import Environment


def small_config() -> TrainingConfig:
    return TrainingConfig(population_size=12, elite_count=2, immigrant_count=1,
                          tournament_size=3, evaluation_seeds=(42, 7), unseen_seeds=(101,), max_steps=120)


class GeneticsTests(unittest.TestCase):
    def test_genome_round_trip_has_exact_parameters_and_no_aliases(self) -> None:
        network = NeuralNetwork(seed=5)
        genome = to_genome(network)
        restored = from_genome(genome)
        np.testing.assert_array_equal(to_genome(restored), genome)
        genome[:] = 0
        self.assertFalse(np.all(restored.w1 == 0))
        for bad in (np.zeros(56), np.full(57, np.nan), np.zeros(57, dtype=complex)):
            with self.assertRaises(ValueError):
                from_genome(bad)

    def test_ga_initialization_is_seeded_and_can_include_existing_network(self) -> None:
        config = small_config()
        a, b = GeneticAlgorithm(config), GeneticAlgorithm(config)
        initial = NeuralNetwork(seed=0)
        pa, pb = a.initialize(initial), b.initialize(initial)
        np.testing.assert_array_equal(pa, pb)
        np.testing.assert_array_equal(pa[0], to_genome(initial))
        self.assertGreater(len(np.unique(pa, axis=0)), 1)

    def test_crossover_only_uses_parent_parameters(self) -> None:
        ga = GeneticAlgorithm(replace(small_config(), crossover_rate=1))
        a, b = np.zeros(57), np.ones(57)
        child = ga.crossover(a, b)
        self.assertTrue(np.all((child == 0) | (child == 1)))
        self.assertTrue(np.any(child == 0) and np.any(child == 1))
        np.testing.assert_array_equal(a, np.zeros(57))

    def test_mutation_changes_weights_and_respects_rate_and_bounds(self) -> None:
        ga = GeneticAlgorithm(replace(small_config(), mutation_rate=1, mutation_scale=0.25, weight_limit=0.1))
        original = np.zeros(57)
        child = ga.mutate(original)
        self.assertTrue(np.any(child != original))
        self.assertTrue(np.all(np.abs(child) <= 0.1))
        np.testing.assert_array_equal(original, np.zeros(57))
        no_mutation = GeneticAlgorithm(replace(small_config(), mutation_rate=0))
        np.testing.assert_array_equal(no_mutation.mutate(original), original)

    def test_elites_are_preserved_without_aliases(self) -> None:
        ga = GeneticAlgorithm(small_config())
        population = ga.initialize()
        fitness = np.arange(len(population), dtype=float)
        before = population.copy()
        children = ga.breed(population, fitness)
        np.testing.assert_array_equal(children[0], before[-1])
        np.testing.assert_array_equal(children[1], before[-2])
        np.testing.assert_array_equal(population, before)
        children[0, 0] += 1
        np.testing.assert_array_equal(population, before)

    def test_tournament_selects_best_of_candidates(self) -> None:
        config = replace(small_config(), tournament_size=12)
        ga = GeneticAlgorithm(config)
        self.assertEqual(ga.select(np.arange(12)), 11)

    def test_invalid_hyperparameters_and_seed_overlap_are_rejected(self) -> None:
        for values in ({"population_size": 1}, {"mutation_rate": 2}, {"max_steps": 0},
                       {"mutation_scale": float("nan")}, {"evaluation_seeds": ()},
                       {"evaluation_seeds": (42, 42)}, {"unseen_seeds": (42,)}, {"training_seed": -1}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                TrainingConfig(**values)


class BatchEvaluationTests(unittest.TestCase):
    def test_batched_birds_match_individual_game_simulations(self) -> None:
        networks = [NeuralNetwork(seed=seed) for seed in (0, 1, 4, 7, 11, 32, 42, 89)]
        population = np.stack([to_genome(network) for network in networks])
        for settings in (DEFAULT_SETTINGS, replace(DEFAULT_SETTINGS, flap_velocity=-900)):
            for seed in (42, 7):
                batch = BatchEpisode(population, seed, 600, settings)
                envs = [Environment(settings, seed) for _ in networks]
                controllers = [NetworkController(network) for network in networks]
                while not batch.done:
                    for env, controller in zip(envs, controllers):
                        env.step(controller(env))
                    batch.step()
                    for index, env in enumerate(envs):
                        with self.subTest(seed=seed, index=index, step=batch.steps):
                            self.assertAlmostEqual(batch.y[index], env.bird.y, places=10)
                            self.assertAlmostEqual(batch.velocity[index], env.bird.velocity, places=10)
                            self.assertAlmostEqual(batch.distances[index], env.distance, places=10)
                            self.assertEqual(int(batch.scores[index]), env.score)
                            self.assertEqual(bool(batch.alive[index]), env.bird.alive)
                            self.assertEqual(str(batch.death_reasons[index]) or None, env.death_reason)

    def test_single_agent_inference_matches_batch_network_math(self) -> None:
        network = NeuralNetwork(seed=4)
        batch = BatchEpisode(to_genome(network)[None, :], 42, 1)
        controller = NetworkController(network)
        controller(Environment(seed=42))
        batch.step()
        self.assertAlmostEqual(batch.outputs[0], controller.last_output, places=14)
        result = batch.results()[0]
        self.assertEqual(result.ticks, 1)
        self.assertTrue(result.step_limit_reached)

    def test_pipe_reward_dominates_default_nonprogress_survival(self) -> None:
        config = TrainingConfig()
        passed = EpisodeResult(42, 1, 100, 300, 1, "pipe", False)
        survival_only = EpisodeResult(42, 0, 1800, 5700, 0, None, True)
        self.assertGreater(passed.fitness(config, DEFAULT_SETTINGS), survival_only.fitness(config, DEFAULT_SETTINGS))

    def test_evaluation_reports_each_seed_and_explicit_failure_metrics(self) -> None:
        results = evaluate_network(NeuralNetwork(seed=0), (101, 202), max_steps=60)
        summary = evaluation_summary(results)
        self.assertEqual(summary["seeds"], [101, 202])
        self.assertEqual(summary["scores"], [0, 0])
        self.assertEqual(summary["zero_score_failure_rate"], 1)
        self.assertEqual(summary["death_rate"], 1)


class TrainerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def test_training_changes_population_and_preserves_best_fitness(self) -> None:
        trainer = Trainer(small_config(), output_dir=self.root)
        initial = trainer.population.copy()
        trainer.run(3, report=lambda _: None)
        self.assertEqual(trainer.generation, 3)
        self.assertFalse(np.array_equal(initial, trainer.population))
        best = [row["all_time_best_fitness"] for row in trainer.history]
        self.assertEqual(best, sorted(best))
        self.assertGreater(trainer.history[-1]["champion_average_flaps"], 0)
        self.assertTrue((self.root / "best.npz").exists())
        np.testing.assert_array_equal(to_genome(NeuralNetwork.load(self.root / "best.npz")), trainer.best_genome)

    def test_resume_matches_uninterrupted_population_rng_and_history_exactly(self) -> None:
        full = Trainer(small_config(), output_dir=self.root / "full")
        full.run(4, report=lambda _: None)
        split = Trainer(small_config(), output_dir=self.root / "split")
        split.run(2, report=lambda _: None)
        resumed = Trainer.load_checkpoint(self.root / "split" / "checkpoint.npz")
        resumed.run(2, report=lambda _: None)
        self.assertEqual(full.history, resumed.history)
        self.assertEqual(full.ga.rng.bit_generator.state, resumed.ga.rng.bit_generator.state)
        np.testing.assert_array_equal(full.population, resumed.population)
        np.testing.assert_array_equal(full.best_genome, resumed.best_genome)

    def test_partial_generation_checkpoint_replays_from_its_start(self) -> None:
        trainer = Trainer(small_config(), output_dir=self.root)
        trainer.advance(5)
        trainer.save_checkpoint()
        restored = Trainer.load_checkpoint(self.root / "checkpoint.npz", output_dir=self.root / "resume")
        self.assertEqual(restored.generation, 0)
        self.assertEqual(restored.episode.steps, 0)
        np.testing.assert_array_equal(restored.population, trainer.population)
        trainer.run(1, report=lambda _: None)
        restored.run(1, report=lambda _: None)
        self.assertEqual(restored.history, trainer.history)

    def test_invalid_checkpoint_and_failed_save_are_handled(self) -> None:
        trainer = Trainer(small_config(), output_dir=self.root)
        trainer.save_checkpoint()
        path = self.root / "checkpoint.npz"
        original = path.read_bytes()
        with patch("ai.trainer.np.savez_compressed", side_effect=OSError("disk error")):
            with self.assertRaises(OSError):
                trainer.save_checkpoint()
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.root.iterdir()), [path])
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(archive["metadata"].item())
            pop = archive["population"].copy()
            champion = archive["champion"].copy()
        meta["generation"] = 4
        np.savez(path, metadata=np.array(json.dumps(meta)), population=pop, champion=champion)
        with self.assertRaises(ValueError):
            Trainer.load_checkpoint(path)
        path.write_bytes(b"corrupt")
        with self.assertRaises(ValueError):
            Trainer.load_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
