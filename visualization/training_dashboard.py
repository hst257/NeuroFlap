"""Observe evolution without influencing any network's inputs or decisions."""

import time
import numpy as np
import pygame
from ai.trainer import Trainer
from game.renderer import CREAM, CORAL, INK, MINT, MUTED, Renderer


class TrainingDashboard:
    def __init__(self, trainer: Trainer, generations: int, speed: int = 16) -> None:
        if generations < 1:
            raise ValueError("generations must be positive")
        if speed not in (1, 2, 4, 8, 16, 32, 64, 128):
            raise ValueError("speed must be a power of two from 1 to 128")
        self.trainer = trainer
        self.generations = generations
        self.target = trainer.generation + generations
        self.speed = speed
        self.paused = False
        self.render_birds = True
        self.best_only = False
        self.visual_time = 0.0
        self.notice = "Weights evolve after each evaluated generation"
        self.reopen()

    def reopen(self) -> None:
        """Return from playback with the live trainer, course and controls intact."""
        pygame.display.init()
        pygame.font.init()
        s = self.trainer.settings
        self.screen = pygame.display.set_mode((s.width, s.height))
        pygame.display.set_caption("NeuroFlap | Learning through evolution")
        self.renderer = Renderer(s)
        self.clock = pygame.time.Clock()
        self.running = True
        self.result = "quit"
        self._reset_rate()

    def _reset_rate(self) -> None:
        self.actual_speed = 0.0
        self._rate_start = time.perf_counter()
        self._rate_ticks = 0
        self._tick_credit = 0.0

    @property
    def finished(self) -> bool:
        return self.trainer.generation >= self.target

    @property
    def displayed_episode(self):
        t = self.trainer
        return t.last_completed_episode if self.finished and t.last_completed_episode is not None else t.episode

    def highlighted_bird(self, episode) -> tuple[int | None, str]:
        """Find the saved champion by weights, even after death or breeding."""
        t, s, c = self.trainer, self.trainer.settings, self.trainer.config
        if self.best_only and t.best_genome is not None:
            matches = np.flatnonzero(np.all(episode.genomes == t.best_genome, axis=1))
            if len(matches):
                index = int(matches[0])
                status = " / OUT THIS COURSE" if not episode.alive[index] else ""
                return index, f"CHAMPION / GEN {t.best_generation}{status}"
            return None, "CHAMPION NOT IN COURSE / W TO WATCH"
        candidates = np.flatnonzero(episode.alive)
        if not len(candidates):
            candidates = np.arange(len(episode.alive))
        fitness = (episode.scores * c.pipe_reward + episode.ticks * s.dt * c.survival_reward
                   + episode.distances / s.pipe_spacing * c.progress_reward)
        index = int(candidates[np.argmax(fitness[candidates])])
        return index, "LIVE LEADER / NO CHAMPION YET" if self.best_only else "POPULATION / LIVE LEADER HIGHLIGHTED"

    def buttons(self) -> list[tuple[str, str, pygame.Rect, bool]]:
        s = self.trainer.settings
        labels = [
            ("more" if self.finished else "pause", "More (N)" if self.finished else "Resume" if self.paused else "Pause", True),
            ("render", "Scene on" if self.render_birds else "Scene off", True),
            ("best", "Best only" if self.best_only else "All birds", True),
            ("save", "Save (S)", True),
            ("slower", "Slower (-)", self.speed > 1),
            ("faster", "Faster (+)", self.speed < 128),
            ("watch", "Watch (W)", self.trainer.best_genome is not None),
            ("quit", "Quit (Esc)", True),
        ]
        width = (s.width - 48) // 4
        return [(action, label, pygame.Rect(20 + (i % 4) * (width + 4),
                                          s.play_height - 116 + (i // 4) * 36, width, 30), enabled)
                for i, (action, label, enabled) in enumerate(labels)]

    def activate(self, action: str) -> None:
        if action == "quit":
            self.running = False
        elif action == "pause":
            self.paused = not self.paused
            self._reset_rate()
        elif action == "faster":
            self.speed = min(128, self.speed * 2)
        elif action == "slower":
            self.speed = max(1, self.speed // 2)
        elif action == "render":
            self.render_birds = not self.render_birds
        elif action == "best":
            self.best_only = not self.best_only
        elif action == "save":
            self.trainer.save_checkpoint()
            if self.trainer.best_genome is not None:
                self.trainer.save_champion()
            self.notice = "Saved checkpoint and available champion"
        elif action == "watch" and self.trainer.best_genome is not None:
            self.result = "watch"
            self.running = False
        elif action == "more" and self.finished:
            self.target += self.generations
            self.paused = False
            self._reset_rate()
            self.notice = "Continuing the same population"

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            self.activate("quit")
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for action, _, rect, enabled in self.buttons():
                if enabled and rect.collidepoint(event.pos):
                    self.activate(action)
                    break
        elif event.type == pygame.KEYDOWN:
            action = {pygame.K_ESCAPE: "quit", pygame.K_SPACE: "pause",
                      pygame.K_EQUALS: "faster", pygame.K_PLUS: "faster", pygame.K_KP_PLUS: "faster",
                      pygame.K_MINUS: "slower", pygame.K_KP_MINUS: "slower",
                      pygame.K_r: "render", pygame.K_b: "best", pygame.K_s: "save",
                      pygame.K_w: "watch", pygame.K_n: "more"}.get(event.key)
            if action:
                self.activate(action)

    def advance(self) -> None:
        if self.paused or self.finished:
            self.actual_speed = 0.0
            return
        deadline = time.perf_counter() + 0.012
        before = self.trainer.generation
        # Nominal speed at a 60 Hz display; physics always uses settings.dt.
        self._tick_credit += self.speed * self.trainer.settings.fps / 60
        budget = int(self._tick_credit)
        self._tick_credit -= budget
        for _ in range(budget):
            self.trainer.advance(stop_generation=self.target)
            self._rate_ticks += 1
            if self.finished or time.perf_counter() >= deadline:
                break
        now = time.perf_counter()
        elapsed = now - self._rate_start
        if elapsed >= 0.5:
            # One shared-course tick, not population_size times the tick count.
            self.actual_speed = self._rate_ticks * self.trainer.settings.dt / elapsed
            self._rate_ticks = 0
            self._rate_start = now
        if self.trainer.generation != before:
            self.notice = f"Gen {self.trainer.generation} evaluated / champion saved"

    def draw(self) -> None:
        t, r, s = self.trainer, self.renderer, self.trainer.settings
        episode = self.displayed_episode
        leader, view_label = self.highlighted_bird(episode)
        r.background(self.screen, self.visual_time, episode.world.distance)
        if self.render_birds:
            r.pipes(self.screen, episode.world)
            if not self.best_only:
                layer = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
                for i in np.flatnonzero(episode.alive):
                    if i != leader:
                        r.bird(layer, s.bird_x, episode.y[i], s.bird_radius,
                               episode.velocity[i], self.visual_time)
                layer.set_alpha(95)
                self.screen.blit(layer, (0, 0))
            if leader is not None:
                r.bird(self.screen, s.bird_x, episode.y[leader], s.bird_radius,
                       episode.velocity[leader], self.visual_time, alive=bool(episode.alive[leader]))
        else:
            scene_status = "COMPLETE" if self.finished else "PAUSED" if self.paused else "TRAINING STILL RUNS"
            r.text(self.screen, f"SCENE OFF / {scene_status}", 22, MUTED,
                   s.width // 2, s.height // 2)
        r.ground(self.screen, episode.world.distance)
        r.panel(self.screen, pygame.Rect(12, 12, s.width - 24, 202))
        cx = s.width // 2
        status = "TRAINING COMPLETE" if self.finished else "PAUSED" if self.paused else "LEARNING / GENETIC ALGORITHM"
        r.text(self.screen, status, 22, MINT, cx, 32)
        generation = t.generation if self.finished else t.generation + 1
        r.text(self.screen, f"GEN {generation} / {self.target}", 32, CREAM, 28, 53, False)
        r.text(self.screen, f"ALIVE {int(episode.alive.sum())}/{t.config.population_size}", 22,
               CORAL, s.width - 176, 59, False)
        row = t.history[-1] if t.history else None
        best = "--" if t.best_fitness is None else f"{t.best_fitness:.1f}"
        average = "--" if row is None else f"{row['average_fitness']:.1f}"
        high = max(int(episode.scores.max()), max((item["highest_score"] for item in t.history), default=0))
        r.text(self.screen, f"ALL-TIME BEST FITNESS  {best}", 22, CREAM, cx, 99)
        r.text(self.screen, f"LAST GEN AVG {average}    HIGH SCORE {high}", 18, CREAM, cx, 124)
        course_index = len(t.config.evaluation_seeds) if self.finished else t.seed_index + 1
        r.text(self.screen, f"COURSE {course_index}/{len(t.config.evaluation_seeds)}    TICK {episode.steps}/{episode.max_steps}",
               18, MUTED, cx, 147)
        speed = 0.0 if self.paused or self.finished else self.actual_speed
        r.text(self.screen, f"ACTUAL {speed:.1f}x    TARGET {self.speed}x", 22, MINT, cx, 173)
        r.text(self.screen, view_label, 18, CORAL if self.best_only else MUTED, cx, 198)
        r.panel(self.screen, pygame.Rect(12, s.play_height - 124, s.width - 24, 114))
        for _, label, rect, enabled in self.buttons():
            pygame.draw.rect(self.screen, (39, 65, 80) if enabled else INK, rect, border_radius=5)
            r.text(self.screen, label, 18, CREAM if enabled else MUTED, rect.centerx, rect.centery)
        r.text(self.screen, "SPACE pause   B best   R scene   +/- speed", 18, MUTED, cx, s.play_height - 39)
        r.text(self.screen, self.notice, 18, MINT, cx, s.play_height - 20)
        if self.finished:
            r.panel(self.screen, pygame.Rect(cx - 170, s.height // 2 - 43, 340, 86))
            r.text(self.screen, "Champion saved to best.npz", 26, CREAM, cx, s.height // 2 - 17)
            r.text(self.screen, "W watch / N train more", 22, MINT, cx, s.height // 2 + 17)

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
