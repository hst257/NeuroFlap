import unittest
from dataclasses import replace

from config import DEFAULT_SETTINGS, Settings
from game.bird import Bird
from game.environment import Environment
from game.pipe import PipePair


class BirdTests(unittest.TestCase):
    def test_gravity_uses_seconds_and_semi_implicit_euler(self) -> None:
        bird = Bird(128, 200, 14)
        bird.update(0.01, DEFAULT_SETTINGS)
        self.assertAlmostEqual(bird.velocity, 14)
        self.assertAlmostEqual(bird.y, 200.14)

    def test_flap_replaces_velocity_and_terminal_speed_is_capped(self) -> None:
        bird = Bird(128, 200, 14, velocity=600)
        bird.flap(DEFAULT_SETTINGS)
        self.assertEqual(bird.velocity, -420)
        bird.update(1, DEFAULT_SETTINGS)
        self.assertEqual(bird.velocity, 680)

    def test_circle_collision_includes_contact_but_not_empty_corner(self) -> None:
        bird = Bird(10, 10, 5)
        self.assertTrue(bird.intersects_rect(15, 8, 20, 20))
        self.assertFalse(bird.intersects_rect(14, 14, 20, 20))
        self.assertTrue(bird.intersects_rect(13, 14, 20, 20))

    def test_dead_bird_cannot_move_or_flap(self) -> None:
        bird = Bird(128, 200, 14, alive=False)
        bird.flap(DEFAULT_SETTINGS)
        bird.update(1, DEFAULT_SETTINGS)
        self.assertEqual((bird.y, bird.velocity), (200, 0))


class EnvironmentTests(unittest.TestCase):
    def snapshot(self, env: Environment) -> tuple:
        return (env.bird.y, env.bird.velocity, env.score, env.ticks,
                env.distance, env.death_reason,
                [(p.x, p.gap_top, p.passed) for p in env.pipes])

    def test_same_seed_and_actions_produce_identical_trajectory(self) -> None:
        a, b = Environment(seed=42), Environment(seed=42)
        for tick in range(900):
            action = tick % 27 == 0
            a.step(action)
            b.step(action)
            self.assertEqual(self.snapshot(a), self.snapshot(b))

    def test_reset_replays_seed_and_different_seed_changes_gaps(self) -> None:
        env = Environment(seed=17)
        initial = self.snapshot(env)
        for _ in range(20):
            env.step()
        env.reset()
        self.assertEqual(initial, self.snapshot(env))
        self.assertNotEqual(env.pipes[0].gap_top, Environment(seed=18).pipes[0].gap_top)

    def test_generated_gaps_respect_margins_and_spacing(self) -> None:
        env = Environment(seed=5)
        for _ in range(100):
            env._spawn_pipe(env.pipes[-1].x + env.settings.pipe_spacing)
        for pipe in env.pipes:
            self.assertGreaterEqual(pipe.gap_top, env.settings.gap_margin)
            self.assertLessEqual(pipe.gap_bottom, env.settings.play_height - env.settings.gap_margin)
        for a, b in zip(env.pipes, env.pipes[1:]):
            self.assertAlmostEqual(b.x - a.x, env.settings.pipe_spacing)

    def test_flap_moves_up_and_no_input_eventually_hits_ground(self) -> None:
        env = Environment(seed=0)
        initial_y = env.bird.y
        env.step(flap=True)
        self.assertLess(env.bird.y, initial_y)
        env.reset()
        for _ in range(180):
            env.step()
        self.assertTrue(env.done)
        self.assertEqual(env.death_reason, "ground")

    def test_ceiling_and_pipe_collision(self) -> None:
        env = Environment(seed=0)
        env.bird.y = env.bird.radius + 1
        env.step(flap=True)
        self.assertEqual(env.death_reason, "ceiling")
        env.reset()
        env.pipes[0].x = env.bird.x
        env.bird.y = env.pipes[0].gap_top - 20
        env.step()
        self.assertEqual(env.death_reason, "pipe")

    def test_score_requires_whole_bird_clear_and_counts_only_once(self) -> None:
        env = Environment(seed=0)
        pipe = env.pipes[0]
        pipe.gap_top = 200
        env.bird.y = 280
        # One pixel beyond the bird's trailing edge is still unpassed.
        pipe.x = env.bird.x - env.bird.radius - pipe.width + 5
        env.step()
        self.assertEqual(env.score, 0)
        env.step()
        self.assertEqual(env.score, 1)
        for _ in range(10):
            env.step()
        self.assertEqual(env.score, 1)

    def test_fast_pipe_cannot_tunnel_through_bird(self) -> None:
        env = Environment(replace(DEFAULT_SETTINGS, pipe_speed=30000), seed=0)
        env.pipes[0].x = 250
        env.pipes[0].gap_top = 400
        env.bird.y = 200
        env.step()
        self.assertEqual(env.death_reason, "pipe")
        self.assertEqual(env.score, 0)

    def test_fast_bird_cannot_tunnel_through_pipe(self) -> None:
        env = Environment(replace(DEFAULT_SETTINGS, flap_velocity=-30000), seed=0)
        env.pipes[0] = PipePair(env.bird.x - 20, 200, 182, 72)
        env.bird.y = 280
        env.step(flap=True)
        self.assertEqual(env.death_reason, "pipe")

    def test_death_freezes_simulation(self) -> None:
        env = Environment(seed=0)
        for _ in range(180):
            env.step()
        frozen = self.snapshot(env)
        env.step(flap=True)
        self.assertEqual(frozen, self.snapshot(env))

    def test_next_pipe_skips_fully_passed_pair(self) -> None:
        env = Environment(seed=0)
        env.pipes[0].x = -100
        self.assertIs(env.next_pipe, env.pipes[1])

    def test_invalid_settings_fail_early(self) -> None:
        for values in ({"pipe_gap": 20}, {"pipe_spacing": 60}, {"gravity": 0},
                       {"pipe_speed": float("nan")}, {"flap_velocity": 50},
                       {"gap_margin": 500}, {"fps": 0}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Settings(**values)


if __name__ == "__main__":
    unittest.main()
