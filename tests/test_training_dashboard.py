import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import pygame
from ai.genetic_algorithm import TrainingConfig
from ai.trainer import Trainer
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
        pygame.time.set_timer(pygame.QUIT, 80, loops=1)
        self.dashboard.run()
        self.assertFalse(pygame.display.get_init())
        restored = Trainer.load_checkpoint(Path(self.folder.name) / "checkpoint.npz")
        self.assertEqual(restored.generation, 2)


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
            imports = subprocess.run([sys.executable, "-c", "import sys; from ai.trainer import Trainer; assert 'pygame' not in sys.modules"],
                                     cwd=root, capture_output=True, text=True)
            self.assertEqual(imports.returncode, 0, imports.stderr)


if __name__ == "__main__":
    unittest.main()
