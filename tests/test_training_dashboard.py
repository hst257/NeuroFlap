import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

from pathlib import Path
import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pygame
from ai.genetic_algorithm import TrainingConfig
from ai.trainer import Trainer
from config import Settings
from visualization.training_dashboard import TrainingDashboard


class DashboardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        config = TrainingConfig(population_size=12, elite_count=1, immigrant_count=1,
                                tournament_size=3, max_steps=4, evaluation_seeds=(42,), unseen_seeds=(101,))
        self.trainer = Trainer(config, output_dir=self.folder.name)
        self.dashboard = TrainingDashboard(self.trainer, 2)

    def tearDown(self) -> None:
        pygame.quit()

    def key(self, key: int) -> None:
        self.dashboard.handle_event(pygame.event.Event(pygame.KEYDOWN, key=key))

    def test_pause_speed_render_and_save_controls(self) -> None:
        self.key(pygame.K_SPACE)
        self.dashboard.advance()
        self.assertEqual(self.trainer.episode.steps, 0)
        self.key(pygame.K_EQUALS)
        self.assertEqual(self.dashboard.speed, 32)
        self.key(pygame.K_MINUS)
        self.assertEqual(self.dashboard.speed, 16)
        self.key(pygame.K_r)
        self.assertFalse(self.dashboard.render_birds)
        self.key(pygame.K_b)
        self.assertTrue(self.dashboard.best_only)
        self.key(pygame.K_s)
        self.assertTrue((Path(self.folder.name) / "checkpoint.npz").exists())
        self.dashboard.draw()

    def test_training_updates_weights_and_watch_control_requires_champion(self) -> None:
        self.key(pygame.K_w)
        self.assertTrue(self.dashboard.running)
        before = self.trainer.population.copy()
        for _ in range(20):
            self.dashboard.advance()
            if self.dashboard.finished:
                break
        self.assertEqual(self.trainer.generation, 2)
        self.assertTrue(self.dashboard.finished)
        self.assertFalse(np.array_equal(before, self.trainer.population))
        self.dashboard.draw()
        self.key(pygame.K_w)
        self.assertEqual(self.dashboard.result, "watch")
        self.assertFalse(self.dashboard.running)

    def test_live_dashboard_quits_and_saves_checkpoint(self) -> None:
        # Finish before scheduling shutdown; filesystem timing must not decide progress.
        self.trainer.advance(8, stop_generation=2)
        pygame.time.set_timer(pygame.QUIT, 80, loops=1)
        self.dashboard.run()
        self.assertFalse(pygame.display.get_init())
        restored = Trainer.load_checkpoint(Path(self.folder.name) / "checkpoint.npz")
        self.assertEqual(restored.generation, 2)

    def test_best_view_tracks_genome_after_death_and_breeding(self) -> None:
        episode = self.trainer.episode
        champion = 7
        self.trainer.best_genome = episode.genomes[champion].copy()
        self.trainer.best_generation = 1
        self.dashboard.best_only = True
        episode.alive[champion] = False
        episode.scores[0] = 100
        self.assertEqual(self.dashboard.highlighted_bird(episode)[0], champion)
        self.assertIn("OUT THIS COURSE", self.dashboard.highlighted_bird(episode)[1])
        self.dashboard.draw()
        self.trainer.last_completed_episode = episode
        self.trainer.generation = self.dashboard.target
        self.trainer.population[:] = 0
        self.assertIs(self.dashboard.displayed_episode, episode)
        self.assertEqual(self.dashboard.highlighted_bird(self.dashboard.displayed_episode)[0], champion)
        self.assertFalse(episode.genomes.flags.writeable)

    def test_best_view_before_champion_and_when_champion_absent(self) -> None:
        self.dashboard.best_only = True
        self.assertIn("NO CHAMPION YET", self.dashboard.highlighted_bird(self.trainer.episode)[1])
        self.trainer.best_genome = np.full(57, 7.0)
        self.assertIsNone(self.dashboard.highlighted_bird(self.trainer.episode)[0])
        self.dashboard.draw()

    def test_render_and_view_controls_do_not_change_evolution(self) -> None:
        reference = Trainer(self.trainer.config, output_dir=Path(self.folder.name) / "reference")
        self.key(pygame.K_r)
        self.key(pygame.K_b)
        for _ in range(8):
            self.trainer.advance(stop_generation=2)
            self.dashboard.draw()
            reference.advance(stop_generation=2)
        np.testing.assert_array_equal(self.trainer.population, reference.population)
        self.assertEqual(self.trainer.history, reference.history)
        self.assertEqual(self.trainer.ga.rng.bit_generator.state, reference.ga.rng.bit_generator.state)

    def test_mouse_controls_and_continuing_completed_session(self) -> None:
        def click(action):
            rect = next(rect for name, _, rect, _ in self.dashboard.buttons() if name == action)
            self.dashboard.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                                                         button=1, pos=rect.center))
        click("watch")
        self.assertTrue(self.dashboard.running)
        click("pause")
        snapshot = self.trainer.population.copy()
        self.dashboard.advance()
        np.testing.assert_array_equal(self.trainer.population, snapshot)
        self.assertEqual(self.trainer.episode.steps, 0)
        click("pause")
        self.trainer.advance(8, stop_generation=2)
        click("more")
        self.assertEqual(self.dashboard.target, 4)
        self.assertFalse(self.dashboard.finished)
        click("watch")
        self.assertEqual(self.dashboard.result, "watch")

    def test_speed_uses_shared_simulation_time_and_configured_fps(self) -> None:
        config = TrainingConfig(population_size=12, elite_count=1, immigrant_count=1,
                                tournament_size=3, max_steps=60, evaluation_seeds=(42,), unseen_seeds=(101,))
        trainer = Trainer(config, settings=Settings(fps=30), output_dir=self.folder.name)
        dashboard = TrainingDashboard(trainer, 2, speed=1)
        with patch("visualization.training_dashboard.time.perf_counter", return_value=100.0):
            dashboard._rate_start = 100.0
            dashboard.advance()
            self.assertEqual(trainer.episode.steps, 0)
            dashboard._rate_start = 99.0
            dashboard.advance()
        self.assertEqual(trainer.episode.steps, 1)
        self.assertAlmostEqual(dashboard.actual_speed, 1 / 30)
        dashboard.activate("pause")
        self.assertEqual(dashboard.actual_speed, 0)

    def test_watch_and_return_preserves_live_training_session(self) -> None:
        from main import run_training
        from game.application import Application
        args = argparse.Namespace(resume=None, training_config=None, population=12,
                                  training_seed=0, max_steps=4, output_dir=self.folder.name,
                                  headless=False, generations=2, train_speed=4, seed=42,
                                  no_auto_restart=False)
        snapshots = []
        def dashboard_run(dashboard):
            if not snapshots:
                dashboard.trainer.advance(13)  # One three-course generation + one live tick.
                snapshots.append((dashboard, dashboard.trainer.episode,
                                  dashboard.trainer.population.copy(),
                                  json.dumps(dashboard.trainer.ga.rng.bit_generator.state)))
                dashboard.activate("watch")
                dashboard.trainer.save_checkpoint()
                pygame.quit()
                return "watch"
            original, episode, population, rng = snapshots[0]
            self.assertIs(dashboard, original)
            self.assertIs(dashboard.trainer.episode, episode)
            self.assertEqual(episode.steps, 1)
            self.assertEqual(dashboard.target, 2)
            self.assertEqual(dashboard.speed, 4)
            np.testing.assert_array_equal(dashboard.trainer.population, population)
            self.assertEqual(json.dumps(dashboard.trainer.ga.rng.bit_generator.state), rng)
            pygame.quit()
            return "quit"
        def watch_run(app):
            self.assertEqual(app.state, "playing")
            self.assertIsNotNone(app.controller)
            app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_t))
            pygame.quit()
        with patch.object(TrainingDashboard, "run", dashboard_run), patch.object(Application, "run", watch_run):
            self.assertEqual(run_training(args), 0)


class TrainingCliTests(unittest.TestCase):
    def test_cli_trains_resumes_and_evaluates_without_pygame(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            def run(*args: str) -> subprocess.CompletedProcess:
                return subprocess.run([sys.executable, "main.py", *args], cwd=root,
                                      capture_output=True, text=True, check=True)
            first = run("--train", "--headless", "--generations", "2", "--population", "12",
                        "--max-steps", "60", "--output-dir", str(output))
            self.assertIn("GEN 002", first.stdout)
            checkpoint = output / "checkpoint.npz"
            self.assertEqual(Trainer.load_checkpoint(checkpoint).generation, 2)
            second = run("--train", "--headless", "--resume", str(checkpoint), "--generations", "2")
            self.assertIn("GEN 004", second.stdout)
            self.assertEqual(Trainer.load_checkpoint(checkpoint).generation, 4)
            evaluated = run("--model", str(output / "best.npz"), "--evaluate", "--max-steps", "60")
            self.assertIn('"average_score"', evaluated.stdout)
            watched = run("--watch", str(output / "best.npz"), "--headless", "--seed", "42", "--steps", "60")
            self.assertIn('"controller": "network"', watched.stdout)
            imports = subprocess.run([sys.executable, "-c", "import sys; from ai.trainer import Trainer; assert 'pygame' not in sys.modules"],
                                     cwd=root, capture_output=True, text=True)
            self.assertEqual(imports.returncode, 0, imports.stderr)


if __name__ == "__main__":
    unittest.main()
