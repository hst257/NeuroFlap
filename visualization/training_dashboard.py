"""A responsive population view: generation progress is actual genetic training."""

import time
import numpy as np
import pygame
from ai.trainer import Trainer
from game.renderer import CREAM, CORAL, INK, MINT, MUTED, Renderer


class TrainingDashboard:
    def __init__(self, trainer: Trainer, generations: int) -> None:
        if generations < 1:
            raise ValueError("generations must be positive")
        pygame.display.init()
        pygame.font.init()
        self.trainer = trainer
        self.target = trainer.generation + generations
        s = trainer.settings
        self.screen = pygame.display.set_mode((s.width, s.height))
        pygame.display.set_caption("NeuroFlap | Learning through evolution")
        self.renderer = Renderer(s)
        self.clock = pygame.time.Clock()
        self.speed = 16
        self.paused = False
        self.render_birds = True
        self.best_only = False
        self.running = True
        self.result = "quit"
        self.visual_time = 0.0
        self.notice = "Weights evolve after every evaluated generation"

    @property
    def finished(self) -> bool:
        return self.trainer.generation >= self.target

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.running = False
            elif event.key == pygame.K_SPACE:
                self.paused = not self.paused
            elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                self.speed = min(128, self.speed * 2)
            elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.speed = max(1, self.speed // 2)
            elif event.key == pygame.K_r:
                self.render_birds = not self.render_birds
            elif event.key == pygame.K_b:
                self.best_only = not self.best_only
            elif event.key == pygame.K_s:
                self.trainer.save_checkpoint()
                if self.trainer.best_genome is not None:
                    self.trainer.save_champion()
                self.notice = "Saved checkpoint and available champion"
            elif event.key == pygame.K_w and self.trainer.best_genome is not None:
                self.result = "watch"
                self.running = False

    def advance(self) -> None:
        if self.paused or self.finished:
            return
        deadline = time.perf_counter() + 0.012
        before = self.trainer.generation
        for _ in range(self.speed):
            self.trainer.advance(stop_generation=self.target)
            if self.finished or time.perf_counter() >= deadline:
                break
        if self.trainer.generation != before:
            row = self.trainer.history[-1]
            self.notice = f"Gen {row['generation']}: saved best / high score {row['highest_score']}"

    def draw(self) -> None:
        t, r, s = self.trainer, self.renderer, self.trainer.settings
        episode = t.last_completed_episode if self.finished and t.last_completed_episode is not None else t.episode
        r.background(self.screen, self.visual_time, episode.world.distance)
        if self.render_birds:
            r.pipes(self.screen, episode.world)
            candidates = np.flatnonzero(episode.alive)
            if len(candidates):
                leader = int(candidates[np.argmax(episode.scores[candidates] * 1000 + episode.distances[candidates])])
                if not self.best_only:
                    layer = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
                    for i in candidates:
                        if i != leader:
                            r.bird(layer, s.bird_x, episode.y[i], s.bird_radius,
                                   episode.velocity[i], self.visual_time)
                    layer.set_alpha(95)
                    self.screen.blit(layer, (0, 0))
                r.bird(self.screen, s.bird_x, episode.y[leader], s.bird_radius,
                       episode.velocity[leader], self.visual_time)
        r.ground(self.screen, episode.world.distance)
        r.panel(self.screen, pygame.Rect(12, 12, s.width - 24, 172))
        cx = s.width // 2
        status = "TRAINING COMPLETE" if self.finished else "PAUSED" if self.paused else "LEARNING / GENETIC ALGORITHM"
        r.text(self.screen, status, 22, MINT, cx, 33)
        generation = t.generation if self.finished else t.generation + 1
        r.text(self.screen, f"GEN {generation} / {self.target}", 32, CREAM, 28, 56, False)
        r.text(self.screen, f"ALIVE {int(episode.alive.sum())}/{t.config.population_size}", 22,
               CORAL, s.width - 176, 62, False)
        row = t.history[-1] if t.history else None
        best = "--" if t.best_fitness is None else f"{t.best_fitness:.1f}"
        average = "--" if row is None else f"{row['average_fitness']:.1f}"
        high = max(int(episode.scores.max()), max((item["highest_score"] for item in t.history), default=0))
        r.text(self.screen, f"BEST FIT {best}    AVG {average}", 22, CREAM, cx, 108)
        course_index = len(t.config.evaluation_seeds) if self.finished else t.seed_index + 1
        r.text(self.screen, f"HIGH SCORE {high}    COURSE {course_index}/{len(t.config.evaluation_seeds)}",
               18, MUTED, cx, 139)
        r.text(self.screen, f"SPEED UP TO {self.speed}x    {'LEADER ONLY' if self.best_only else 'POPULATION VIEW'}",
               18, MINT, cx, 163)
        r.panel(self.screen, pygame.Rect(12, s.play_height - 89, s.width - 24, 78))
        r.text(self.screen, "SPACE pause   +/- speed   B leader   R render", 18, CREAM, cx, s.play_height - 67)
        r.text(self.screen, "S save   W watch champion   ESC save & quit", 18, MINT, cx, s.play_height - 44)
        r.text(self.screen, self.notice, 18, MUTED, cx, s.play_height - 22)
        if self.finished:
            r.panel(self.screen, pygame.Rect(cx - 179, s.height // 2 - 46, 358, 100))
            r.text(self.screen, "Champion saved to best.npz", 26, CREAM, cx, s.height // 2 - 15)
            r.text(self.screen, "Press W to watch it play", 22, MINT, cx, s.height // 2 + 23)

    def run(self) -> str:
        try:
            while self.running:
                self.visual_time += self.clock.tick(60) / 1000.0
                for event in pygame.event.get():
                    self.handle_event(event)
                if not self.running:
                    break
                self.advance()
                self.draw()
                pygame.display.flip()
            return self.result
        finally:
            try:
                self.trainer.save_checkpoint()
                if self.trainer.best_genome is not None:
                    self.trainer.save_champion()
            finally:
                pygame.quit()
