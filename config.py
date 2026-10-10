"""Tunable game settings. Distances are pixels; time is seconds."""

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class Settings:
    width: int = 480
    height: int = 720
    ground_height: int = 60
    fps: int = 60
    bird_x: float = 128.0
    bird_radius: float = 14.0
    gravity: float = 1400.0
    flap_velocity: float = -420.0
    terminal_velocity: float = 680.0
    pipe_width: float = 72.0
    pipe_gap: float = 182.0
    pipe_spacing: float = 245.0
    pipe_speed: float = 190.0
    gap_margin: float = 72.0
    first_pipe_offset: float = 100.0
    network_restart_delay: float = 1.5

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.width < 320 or self.height < 480 or self.fps < 1:
            raise ValueError("Use a window of at least 320 x 480 and a positive FPS")
        if not 0 < self.ground_height < self.height:
            raise ValueError("ground_height must fit inside the window")
        if any(value <= 0 for value in (
            self.bird_radius, self.gravity, self.terminal_velocity,
            self.pipe_width, self.pipe_gap, self.pipe_spacing, self.pipe_speed,
        )):
            raise ValueError("Physics sizes and speeds must be positive")
        if self.flap_velocity >= 0:
            raise ValueError("flap_velocity must be negative (upward)")
        if self.gap_margin < 0 or self.first_pipe_offset < 0:
            raise ValueError("Pipe margins must be nonnegative")
        if self.network_restart_delay < 0:
            raise ValueError("network_restart_delay must be nonnegative")
        if self.pipe_gap <= 2 * self.bird_radius:
            raise ValueError("The bird must fit through the gap")
        if self.pipe_gap + 2 * self.gap_margin >= self.play_height:
            raise ValueError("Pipe gap and margins must fit above the ground")
        if self.pipe_spacing <= self.pipe_width:
            raise ValueError("Pipe spacing must exceed pipe width")
        if not self.bird_radius < self.bird_x < self.width - self.bird_radius:
            raise ValueError("bird_x must fit inside the window")
        if self.bird_radius * 2 >= self.play_height:
            raise ValueError("The bird must fit above the ground")

    @property
    def play_height(self) -> int:
        return self.height - self.ground_height

    @property
    def dt(self) -> float:
        return 1.0 / self.fps


DEFAULT_SETTINGS = Settings()
