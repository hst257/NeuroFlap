"""Interactive state machine and fixed-step clock; physics lives in Environment."""

import pygame
from collections.abc import Callable
from config import DEFAULT_SETTINGS, Settings
from .environment import Environment
from .renderer import Renderer


class Application:
    def __init__(self, settings: Settings = DEFAULT_SETTINGS, seed: int | None = None,
                 controller: Callable[[Environment], bool] | None = None,
                 auto_restart: bool = True) -> None:
        pygame.display.init()
        pygame.font.init()
        self.settings = settings
        self.screen = pygame.display.set_mode((settings.width, settings.height))
        pygame.display.set_caption("NeuroFlap | Network pilot" if controller is not None else "NeuroFlap | Human pilot")
        self.renderer = Renderer(settings)
        self.env = Environment(settings, seed)
        self.controller = controller
        self.auto_restart = auto_restart
        self.clock = pygame.time.Clock()
        self.state = "menu"
        self.running = True
        self.training_requested = False
        self.best = 0
        self.focus = 0
        self.accumulator = 0.0
        self.flap_pending = False
        self.visual_time = 0.0
        self.attempt = 0
        self.flap_count = 0
        self.last_network_output: float | None = None
        self.restart_remaining: float | None = None

    def start(self) -> None:
        self.env.reset()
        self.state = "playing" if self.controller is not None else "ready"
        self.focus = 0
        self.accumulator = 0.0
        self.flap_pending = False
        self.attempt += 1
        self.flap_count = 0
        self.last_network_output = None
        self.restart_remaining = None

    def menu(self) -> None:
        self.state = "menu"
        self.focus = 0
        self.accumulator = 0.0
        self.flap_pending = False
        self.restart_remaining = None

    def activate(self, index: int) -> None:
        if self.state == "menu":
            if index == 0:
                self.start()
            else:
                self.running = False
        elif self.state == "gameover":
            self.start() if index == 0 else self.menu()

    def flap(self) -> None:
        if self.state == "gameover":
            self.start()
        if self.state == "ready":
            self.state = "playing"
            self.accumulator = 0.0
        if self.state == "playing" and self.controller is None:
            self.flap_pending = True

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_t:
                self.training_requested = True
                self.running = False
            elif event.key == pygame.K_ESCAPE:
                if self.state == "menu":
                    self.running = False
                else:
                    self.menu()
            elif event.key in (pygame.K_TAB, pygame.K_UP, pygame.K_DOWN):
                self.focus = (self.focus + 1) % 2
            elif event.key == pygame.K_RETURN:
                if self.state in ("menu", "gameover"):
                    self.activate(self.focus)
                else:
                    self.flap()
            elif event.key == pygame.K_SPACE:
                self.start() if self.state == "menu" else self.flap()
            elif event.key == pygame.K_r and self.state in ("playing", "ready", "gameover"):
                self.start()
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.state in ("menu", "gameover"):
                for index, (_, rect) in enumerate(self.renderer.buttons(self.state)):
                    if rect.collidepoint(event.pos):
                        self.activate(index)
                        break
            else:
                self.flap()

    def advance(self, frame_seconds: float) -> None:
        """Bound catch-up after a stalled window; never use variable physics dt."""
        self.visual_time += frame_seconds
        if self.state == "gameover" and self.restart_remaining is not None:
            self.restart_remaining = max(0.0, self.restart_remaining - frame_seconds)
            if self.restart_remaining == 0.0:
                self.start()
            return
        if self.state != "playing":
            self.accumulator = 0.0
            return
        self.accumulator += min(frame_seconds, 0.1)
        while self.accumulator >= self.settings.dt:
            action = self.controller(self.env) if self.controller is not None else self.flap_pending
            if self.controller is not None:
                self.last_network_output = getattr(self.controller, "last_output", None)
            self.flap_count += int(action)
            self.env.step(action)
            self.flap_pending = False
            self.accumulator -= self.settings.dt
            self.best = max(self.best, self.env.score)
            if self.env.done:
                self.state = "gameover"
                self.focus = 0
                self.accumulator = 0.0
                if self.controller is not None and self.auto_restart:
                    self.restart_remaining = self.settings.network_restart_delay
                break

    def run(self) -> None:
        try:
            while self.running:
                frame_seconds = self.clock.tick(self.settings.fps) / 1000.0
                for event in pygame.event.get():
                    self.handle_event(event)
                self.advance(frame_seconds)
                self.renderer.draw(self.screen, self.env, self.state, self.visual_time,
                                   self.best, pygame.mouse.get_pos(), self.focus,
                                   network_controlled=self.controller is not None,
                                   network_output=self.last_network_output,
                                   flap_count=self.flap_count, attempt=self.attempt,
                                   restart_remaining=self.restart_remaining)
                pygame.display.flip()
        finally:
            pygame.quit()
