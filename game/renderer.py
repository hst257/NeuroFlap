"""Original artwork made entirely with Pygame primitives and bundled fonts."""

import math
import random
import pygame
from config import Settings
from .environment import Environment

INK = (22, 39, 56)
MUTED = (121, 161, 172)
CREAM = (243, 244, 219)
MINT = (126, 227, 183)
CORAL = (255, 164, 126)


class Renderer:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.fonts = {size: pygame.font.Font(None, size)
                      for size in (18, 22, 26, 32, 40, 58, 76)}
        self.sky = pygame.Surface((settings.width, settings.height))
        for y in range(settings.height):
            blend = y / settings.height
            color = tuple(round(a + (b - a) * blend)
                          for a, b in zip((22, 40, 65), (60, 105, 116)))
            pygame.draw.line(self.sky, color, (0, y), (settings.width, y))
        rng = random.Random(17)
        self.stars = [(rng.randrange(settings.width), rng.randrange(settings.play_height // 2),
                       rng.choice((1, 1, 2))) for _ in range(34)]

    def text(self, screen: pygame.Surface, text: str, size: int,
             color: tuple[int, int, int], x: float, y: float,
             centered: bool = True) -> None:
        surface = self.fonts[size].render(text, True, color)
        rect = surface.get_rect(center=(round(x), round(y))) if centered else surface.get_rect(topleft=(round(x), round(y)))
        screen.blit(surface, rect)

    def buttons(self, state: str) -> list[tuple[str, pygame.Rect]]:
        w, h = self.s.width, self.s.height
        if state == "menu":
            return [("PLAY", pygame.Rect(w // 2 - 122, h // 2 + 56, 244, 52)),
                    ("QUIT", pygame.Rect(w // 2 - 122, h // 2 + 120, 244, 44))]
        if state == "gameover":
            return [("TRY AGAIN", pygame.Rect(w // 2 - 122, h // 2 + 50, 244, 52)),
                    ("MENU", pygame.Rect(w // 2 - 122, h // 2 + 114, 244, 44))]
        return []

    def button(self, screen: pygame.Surface, rect: pygame.Rect, label: str,
               primary: bool, selected: bool) -> None:
        shadow = rect.move(0, 5)
        pygame.draw.rect(screen, (13, 30, 44), shadow, border_radius=9)
        color = (163, 241, 202) if selected and primary else MINT if primary else (39, 65, 80)
        pygame.draw.rect(screen, color, rect, border_radius=9)
        if selected:
            pygame.draw.rect(screen, CREAM, rect, 2, border_radius=9)
        self.text(screen, label, 26, INK if primary else CREAM, rect.centerx, rect.centery + 1)

    def background(self, screen: pygame.Surface, time: float, distance: float) -> None:
        s = self.s
        screen.blit(self.sky, (0, 0))
        for x, y, r in self.stars:
            pygame.draw.rect(screen, (101, 142, 161), (x, y, r, r))
        pygame.draw.circle(screen, (179, 205, 196), (s.width - 72, 82), 22)
        pygame.draw.circle(screen, (29, 50, 74), (s.width - 64, 75), 21)
        for i in range(4):
            x = int((i * 181 - time * (8 + i * 2)) % (s.width + 150)) - 100
            y = 185 + (i % 3) * 62
            pygame.draw.rect(screen, (66, 104, 122), (x, y, 86, 10))
            pygame.draw.rect(screen, (66, 104, 122), (x + 16, y - 10, 42, 12))
        for i in range(-1, s.width // 90 + 2):
            x = i * 90 - int(distance * 0.13) % 90
            pygame.draw.polygon(screen, (38, 72, 87),
                                [(x - 10, s.play_height), (x + 38, s.play_height - 94),
                                 (x + 100, s.play_height)])
            pygame.draw.polygon(screen, (44, 82, 91),
                                [(x + 30, s.play_height), (x + 86, s.play_height - 62),
                                 (x + 138, s.play_height)])

    def ground(self, screen: pygame.Surface, distance: float) -> None:
        s = self.s
        pygame.draw.rect(screen, (28, 48, 59), (0, s.play_height, s.width, s.ground_height))
        pygame.draw.rect(screen, MINT, (0, s.play_height, s.width, 5))
        pygame.draw.rect(screen, (68, 132, 124), (0, s.play_height + 5, s.width, 6))
        offset = int(distance) % 28
        for x in range(-28, s.width + 28, 28):
            pygame.draw.rect(screen, (42, 67, 74), (x - offset, s.play_height + 19, 15, 4))
            pygame.draw.rect(screen, (35, 59, 68), (x - offset + 13, s.play_height + 38, 15, 4))

    def pipes(self, screen: pygame.Surface, env: Environment) -> None:
        for pipe in env.pipes:
            x, w = round(pipe.x), round(pipe.width)
            for y, h, cap_y in ((0, round(pipe.gap_top), round(pipe.gap_top) - 20),
                                (round(pipe.gap_bottom), self.s.play_height - round(pipe.gap_bottom), round(pipe.gap_bottom))):
                pygame.draw.rect(screen, (21, 56, 65), (x, y, w, h))
                pygame.draw.rect(screen, (72, 150, 134), (x + 4, y, w - 8, h))
                pygame.draw.rect(screen, (109, 192, 158), (x + 8, y, 8, h))
                pygame.draw.rect(screen, (48, 111, 110), (x + w - 15, y, 9, h))
                pygame.draw.rect(screen, (21, 56, 65), (x, cap_y, w, 20))
                pygame.draw.rect(screen, MINT, (x + 3, cap_y + 3, w - 6, 5))
                pygame.draw.rect(screen, (77, 157, 137), (x + 3, cap_y + 8, w - 6, 9))

    def bird(self, screen: pygame.Surface, x: float, y: float, radius: float,
             velocity: float, time: float, alive: bool = True) -> None:
        r = max(8, round(radius))
        sprite = pygame.Surface((r * 4, r * 4), pygame.SRCALPHA)
        cx = cy = r * 2
        points = [(cx - r, cy - r // 3), (cx - r // 2, cy - r + 1),
                  (cx + r // 2, cy - r + 1), (cx + r - 1, cy - r // 3),
                  (cx + r - 1, cy + r // 3), (cx + r // 2, cy + r - 1),
                  (cx - r // 2, cy + r - 1), (cx - r, cy + r // 3)]
        pygame.draw.polygon(sprite, INK, points)
        pygame.draw.polygon(sprite, CORAL, [(cx + (px - cx) * 0.84, cy + (py - cy) * 0.84)
                                          for px, py in points])
        wing_y = cy + round(math.sin(time * 22) * r * 0.14) if alive else cy
        pygame.draw.rect(sprite, (210, 104, 87), (cx - r * 0.7, wing_y, r * 0.65, r * 0.35))
        pygame.draw.rect(sprite, CREAM, (cx + r * 0.1, cy - r * 0.55, r * 0.4, r * 0.4))
        pygame.draw.rect(sprite, INK, (cx + r * 0.32, cy - r * 0.5, max(2, r * 0.15), max(2, r * 0.2)))
        pygame.draw.rect(sprite, (255, 220, 150), (cx + r * 0.55, cy, r * 0.3, r * 0.22))
        angle = max(-70, min(25, -velocity * 0.075))
        rotated = pygame.transform.rotate(sprite, angle)
        screen.blit(rotated, rotated.get_rect(center=(round(x), round(y))))

    def panel(self, screen: pygame.Surface, rect: pygame.Rect) -> None:
        pygame.draw.rect(screen, (13, 30, 44), rect.move(0, 8), border_radius=16)
        pygame.draw.rect(screen, INK, rect, border_radius=16)
        pygame.draw.rect(screen, (71, 110, 121), rect, 1, border_radius=16)

    def draw(self, screen: pygame.Surface, env: Environment, state: str,
             time: float, best: int, mouse: tuple[int, int], focus: int = 0,
             network_controlled: bool = False, network_output: float | None = None,
             flap_count: int = 0, attempt: int = 0,
             restart_remaining: float | None = None) -> None:
        s = self.s
        cx, cy = s.width // 2, s.height // 2
        self.background(screen, time, env.distance)
        if state != "menu":
            self.pipes(screen, env)
            bob = math.sin(time * 4) * 5 if state == "ready" else 0
            self.bird(screen, env.bird.x, env.bird.y + bob, env.bird.radius,
                      env.bird.velocity, time, env.bird.alive)
        self.ground(screen, env.distance)
        if state == "menu":
            self.text(screen, "A LITTLE BIRD. A BIG LEAP.", 18, MINT, cx, cy - 236)
            self.text(screen, "NeuroFlap", 76, (12, 29, 44), cx + 2, cy - 185 + 4)
            self.text(screen, "NeuroFlap", 76, CREAM, cx, cy - 185)
            self.text(screen, "Find your rhythm. Chase the gap.", 22, MUTED, cx, cy - 132)
            self.bird(screen, cx, cy - 45 + math.sin(time * 3) * 8, 38, 0, time)
            hint = "WATCH MODE / FIXED WEIGHTS" if network_controlled else "SPACE / CLICK TO FLAP"
            pilot = "NETWORK WATCH MODE" if network_controlled else "HUMAN PILOT"
            self.text(screen, hint, 22, CREAM, cx, cy + 23)
            self.text(screen, pilot, 18, MUTED, cx, s.play_height - 36)
            self.text(screen, "T   START LEARNING", 22, MINT, cx, cy + 188)
        elif state in ("playing", "ready"):
            self.text(screen, str(env.score), 58, INK, cx + 2, 61 + 3)
            self.text(screen, str(env.score), 58, CREAM, cx, 61)
            self.text(screen, f"BEST {best:02d}", 18, MINT, 22, 24, False)
            self.text(screen, "ESC  MENU", 18, MUTED, s.width - 97, 24, False)
            if network_controlled:
                self.text(screen, "WATCH ONLY / T TO TRAIN", 18, MINT, cx, 101)
                output = "--" if network_output is None else f"{network_output:.3f}"
                action = "FLAP" if network_output is not None and network_output > 0.5 else "WAIT"
                self.text(screen, f"OUTPUT {output}   /   {action}   /   FLAPS {flap_count}",
                          18, CREAM, cx, 125)
                self.text(screen, f"ATTEMPT {attempt}   /   FLAP WHEN OUTPUT > 0.5", 18, MUTED, cx, 148)
            else:
                self.text(screen, "HUMAN PILOT", 18, MUTED, cx, 101)
            if state == "ready":
                self.panel(screen, pygame.Rect(cx - 156, cy + 27, 312, 105))
                self.text(screen, "READY TO FLY?", 32, CREAM, cx, cy + 60)
                self.text(screen, "Press SPACE or click", 22, MINT, cx, cy + 96)
        elif state == "gameover":
            veil = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
            veil.fill((10, 22, 35, 125))
            screen.blit(veil, (0, 0))
            self.panel(screen, pygame.Rect(cx - 184, cy - 176, 368, 364))
            self.text(screen, "FLIGHT COMPLETE", 18, CORAL, cx, cy - 145)
            self.text(screen, "Keep your wings up.", 32, CREAM, cx, cy - 106)
            self.text(screen, "SCORE", 18, MUTED, cx - 75, cy - 58)
            self.text(screen, "BEST", 18, MUTED, cx + 75, cy - 58)
            self.text(screen, str(env.score), 58, CREAM, cx - 75, cy - 19)
            self.text(screen, str(best), 58, MINT, cx + 75, cy - 19)
            self.text(screen, "T TRAIN    SPACE / R RETRY    ESC MENU", 18, MUTED, cx, cy + 221)
            if network_controlled:
                output = "--" if network_output is None else f"{network_output:.3f}"
                self.text(screen, f"LAST OUTPUT {output}   /   FLAPS {flap_count}",
                          18, CREAM, cx, cy - 192)
                if restart_remaining is not None:
                    self.text(screen, f"NEXT ATTEMPT IN {restart_remaining:.1f}s", 22, MINT, cx, cy + 204)
        for index, (label, rect) in enumerate(self.buttons(state)):
            self.button(screen, rect, label, index == 0, index == focus or rect.collidepoint(mouse))
