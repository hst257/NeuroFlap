"""Pipe pairs share a gap and a scoring flag."""

from dataclasses import dataclass
from .bird import Bird
from config import Settings


@dataclass(slots=True)
class PipePair:
    x: float
    gap_top: float
    gap_size: float
    width: float
    passed: bool = False

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def gap_bottom(self) -> float:
        return self.gap_top + self.gap_size

    def update(self, dt: float, settings: Settings) -> None:
        self.x -= settings.pipe_speed * dt

    def collides(self, bird: Bird, play_height: float) -> bool:
        return (bird.intersects_rect(self.x, 0.0, self.width, self.gap_top)
                or bird.intersects_rect(self.x, self.gap_bottom, self.width,
                                        play_height - self.gap_bottom))
