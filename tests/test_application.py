"""Exercise real Pygame input and draw paths using SDL's offscreen driver."""

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import unittest
import pygame
from game.application import Application


class ApplicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = Application(seed=42)

    def tearDown(self) -> None:
        pygame.quit()

    def key(self, key: int) -> None:
        self.app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=key))

    def test_keyboard_play_flap_death_restart_and_menu(self) -> None:
        self.key(pygame.K_RETURN)
        self.assertEqual(self.app.state, "ready")
        self.key(pygame.K_SPACE)
        self.app.advance(1 / 60)
        self.assertEqual(self.app.state, "playing")
        self.assertLess(self.app.env.bird.velocity, 0)
        for _ in range(180):
            self.app.advance(1 / 60)
        self.assertEqual(self.app.state, "gameover")
        self.key(pygame.K_SPACE)
        self.app.advance(1 / 60)
        self.assertEqual(self.app.state, "playing")
        self.assertEqual(self.app.env.score, 0)
        self.key(pygame.K_ESCAPE)
        self.assertEqual(self.app.state, "menu")

    def test_click_buttons_and_quit(self) -> None:
        play_rect = self.app.renderer.buttons("menu")[0][1]
        self.app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=play_rect.center))
        self.assertEqual(self.app.state, "ready")
        self.app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(100, 200)))
        self.assertEqual(self.app.state, "playing")
        self.key(pygame.K_ESCAPE)
        self.key(pygame.K_DOWN)
        self.key(pygame.K_RETURN)
        self.assertFalse(self.app.running)

    def test_queued_flap_is_not_lost_between_ticks(self) -> None:
        self.app.start()
        self.app.flap()
        self.app.advance(0.001)
        self.assertTrue(self.app.flap_pending)
        self.app.advance(1 / 60)
        self.assertFalse(self.app.flap_pending)
        self.assertLess(self.app.env.bird.velocity, 0)

    def test_render_all_states_and_bound_stalled_frame(self) -> None:
        for state in ("menu", "ready", "playing", "gameover"):
            self.app.renderer.draw(self.app.screen, self.app.env, state, 2.0, 5, (0, 0))
            pygame.display.flip()
        self.app.start()
        self.app.flap()
        self.app.advance(30)
        self.assertLessEqual(self.app.env.ticks, 6)

    def test_live_loop_honors_window_close_and_releases_display(self) -> None:
        pygame.time.set_timer(pygame.QUIT, 60, loops=1)
        self.app.run()
        self.assertFalse(self.app.running)
        self.assertFalse(pygame.display.get_init())

    def test_network_decides_each_tick_and_manual_flaps_do_not_override(self) -> None:
        from ai.controller import NetworkController
        from ai.neural_network import NeuralNetwork, PARAMETER_SHAPES
        import numpy as np
        network = NeuralNetwork(parameters={name: np.zeros(shape) for name, shape in PARAMETER_SHAPES.items()})
        self.app.controller = NetworkController(network)
        self.app.start()
        self.assertEqual(self.app.state, "playing")
        self.key(pygame.K_SPACE)
        self.assertFalse(self.app.flap_pending)
        self.app.advance(1 / 60)
        self.assertGreater(self.app.env.bird.velocity, 0)
        self.assertEqual(self.app.last_network_output, 0.5)
        self.assertEqual(self.app.flap_count, 0)
        network.b2[0] = 1
        self.app.advance(1 / 60)
        self.assertLess(self.app.env.bird.velocity, 0)
        self.assertGreater(self.app.last_network_output, 0.5)
        self.assertEqual(self.app.flap_count, 1)
        self.app.renderer.draw(self.app.screen, self.app.env, "playing", 1, 0, (0, 0),
                               network_controlled=True)
        self.key(pygame.K_r)
        self.assertEqual(self.app.state, "playing")
        self.assertEqual(self.app.env.ticks, 0)
        self.assertEqual(self.app.flap_count, 0)
        self.assertIsNone(self.app.last_network_output)

    def lose_network_flight(self) -> None:
        self.app.controller = lambda env: False
        self.app.start()
        for _ in range(180):
            self.app.advance(1 / 60)
            if self.app.state == "gameover":
                break
        self.assertEqual(self.app.state, "gameover")

    def test_network_restarts_after_countdown_and_resets_attempt_statistics(self) -> None:
        self.lose_network_flight()
        self.assertEqual(self.app.attempt, 1)
        self.assertEqual(self.app.restart_remaining, 1.5)
        self.app.renderer.draw(self.app.screen, self.app.env, "gameover", 1, 0, (0, 0),
                               network_controlled=True, network_output=0.42,
                               restart_remaining=1.5, flap_count=0, attempt=1)
        self.app.advance(0.5)
        self.assertEqual(self.app.state, "gameover")
        self.assertEqual(self.app.restart_remaining, 1.0)
        self.app.advance(1.0)
        self.assertEqual(self.app.state, "playing")
        self.assertEqual(self.app.attempt, 2)
        self.assertEqual(self.app.env.ticks, 0)
        self.assertFalse(self.app.env.done)
        self.assertIsNone(self.app.restart_remaining)
        self.assertIsNone(self.app.last_network_output)
        self.assertEqual(self.app.flap_count, 0)

    def test_escape_cancels_pending_network_restart(self) -> None:
        self.lose_network_flight()
        self.key(pygame.K_ESCAPE)
        self.app.advance(10)
        self.assertEqual(self.app.state, "menu")
        self.assertIsNone(self.app.restart_remaining)
        self.assertEqual(self.app.attempt, 1)

    def test_manual_retry_cancels_old_timer(self) -> None:
        self.lose_network_flight()
        self.key(pygame.K_r)
        self.assertEqual(self.app.state, "playing")
        self.assertIsNone(self.app.restart_remaining)
        self.app.advance(0.05)
        self.assertEqual(self.app.attempt, 2)
        self.assertGreater(self.app.env.ticks, 0)

    def test_network_auto_restart_can_be_disabled(self) -> None:
        self.app.auto_restart = False
        self.lose_network_flight()
        self.app.advance(10)
        self.assertEqual(self.app.state, "gameover")
        self.assertIsNone(self.app.restart_remaining)

    def test_human_flight_stays_on_gameover_screen(self) -> None:
        self.app.start()
        self.app.flap()
        for _ in range(180):
            self.app.advance(1 / 60)
        self.assertEqual(self.app.state, "gameover")
        self.app.advance(10)
        self.assertEqual(self.app.state, "gameover")
        self.assertIsNone(self.app.restart_remaining)

    def test_untrained_seed_zero_is_evaluated_each_tick_even_with_zero_flaps(self) -> None:
        from ai.controller import NetworkController
        from ai.neural_network import NeuralNetwork
        self.app.controller = NetworkController(NeuralNetwork(seed=0))
        self.app.start()
        outputs = []
        for _ in range(600):
            self.app.advance(1 / 60)
            outputs.append(self.app.last_network_output)
            if self.app.state == "gameover":
                break
        self.assertEqual(self.app.env.death_reason, "ground")
        self.assertEqual(self.app.env.ticks, 46)
        self.assertEqual(len(outputs), self.app.env.ticks)
        self.assertTrue(all(output is not None and output < 0.5 for output in outputs))
        self.assertEqual(self.app.flap_count, 0)
        self.assertIsNotNone(self.app.restart_remaining)

    def test_t_requests_training_from_watch_mode(self) -> None:
        self.app.controller = lambda env: False
        self.app.start()
        self.key(pygame.K_t)
        self.assertTrue(self.app.training_requested)
        self.assertFalse(self.app.running)


if __name__ == "__main__":
    unittest.main()
