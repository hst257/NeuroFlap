"""Bird motion and exact circle-versus-rectangle geometry; no Pygame needed."""

from dataclasses import dataclass
from config import Settings


@dataclass(slots=True)
class Bird:
    x: float
    y: float
    radius: float
    velocity: float = 0.0
    alive: bool = True

    def flap(self, settings: Settings) -> None:
        if self.alive:
            self.velocity = settings.flap_velocity

    def update(self, dt: float, settings: Settings) -> None:
        if not self.alive:
            return
        self.velocity = min(self.velocity + settings.gravity * dt,
                            settings.terminal_velocity)
        self.y += self.velocity * dt

    def intersects_rect(self, x: float, y: float, width: float, height: float) -> bool:
        closest_x = max(x, min(self.x, x + width))
        closest_y = max(y, min(self.y, y + height))
        return ((self.x - closest_x) ** 2 + (self.y - closest_y) ** 2
                <= self.radius ** 2)
