"""Deterministic fixed-tick environment, independent of clocks and graphics."""

import math
import random
from config import DEFAULT_SETTINGS, Settings
from .bird import Bird
from .pipe import PipePair


class Environment:
    def __init__(self, settings: Settings = DEFAULT_SETTINGS,
                 seed: int | None = None) -> None:
        self.settings = settings
        self.seed = seed
        self.reset()

    def reset(self) -> None:
        """Replay the same course for a supplied seed; None makes a fresh course."""
        self.rng = random.Random(self.seed)
        self.bird = Bird(self.settings.bird_x, self.settings.play_height * 0.45,
                         self.settings.bird_radius)
        self.pipes: list[PipePair] = []
        self.score = 0
        self.ticks = 0
        self.distance = 0.0
        self.death_reason: str | None = None
        self._spawn_pipe(self.settings.width + self.settings.first_pipe_offset)
        self._fill_pipes()

    @property
    def elapsed(self) -> float:
        return self.ticks * self.settings.dt

    @property
    def done(self) -> bool:
        return not self.bird.alive

    @property
    def next_pipe(self) -> PipePair:
        return next(pipe for pipe in self.pipes if pipe.right >= self.bird.x - self.bird.radius)

    def _spawn_pipe(self, x: float) -> None:
        s = self.settings
        top = self.rng.uniform(s.gap_margin, s.play_height - s.gap_margin - s.pipe_gap)
        self.pipes.append(PipePair(x, top, s.pipe_gap, s.pipe_width))

    def _fill_pipes(self) -> None:
        while self.pipes[-1].x < self.settings.width + self.settings.pipe_spacing:
            self._spawn_pipe(self.pipes[-1].x + self.settings.pipe_spacing)

    def step(self, flap: bool = False) -> None:
        """Advance ONE fixed tick. Accelerate by calling more often, not larger dt.

        Substeps limit relative motion to half a bird radius, preventing obstacles
        being skipped even when configurable speeds are much faster than normal.
        """
        if self.done:
            return
        s = self.settings
        if flap:
            self.bird.flap(s)
        speed_bound = max(abs(s.flap_velocity), s.terminal_velocity) + s.gravity * s.dt
        substeps = max(1, math.ceil((speed_bound + s.pipe_speed) * s.dt
                                    / (self.bird.radius * 0.5)))
        dt = s.dt / substeps
        self.ticks += 1
        for _ in range(substeps):
            self.bird.update(dt, s)
            for pipe in self.pipes:
                pipe.update(dt, s)
            self.distance += s.pipe_speed * dt
            self._fill_pipes()
            if self.bird.y - self.bird.radius <= 0:
                self.death_reason = "ceiling"
            elif self.bird.y + self.bird.radius >= s.play_height:
                self.death_reason = "ground"
            elif any(pipe.collides(self.bird, s.play_height) for pipe in self.pipes):
                self.death_reason = "pipe"
            if self.death_reason:
                self.bird.alive = False
                break
            for pipe in self.pipes:
                if not pipe.passed and pipe.right < self.bird.x - self.bird.radius:
                    pipe.passed = True
                    self.score += 1
            self.pipes = [pipe for pipe in self.pipes if pipe.right >= 0]
