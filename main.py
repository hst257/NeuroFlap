"""Launch human play, inspect an untrained model, or run headless inference."""

import argparse
import json
from pathlib import Path
import sys


def run_training(args: argparse.Namespace, initial=None) -> int:
    """Start evolution explicitly or via T in the interactive game."""
    from ai.genetic_algorithm import TrainingConfig
    from ai.trainer import Trainer
    from ai.evaluator import evaluate_network, evaluation_summary
    if args.resume:
        if any(value is not None for value in (args.training_config, args.population, args.training_seed, args.max_steps)):
            raise ValueError("Resume uses checkpoint settings; do not override training configuration")
        trainer = Trainer.load_checkpoint(args.resume, output_dir=args.output_dir)
    else:
        values = {}
        if args.training_config:
            values = json.loads(Path(args.training_config).read_text(encoding="utf-8"))
            if not isinstance(values, dict):
                raise ValueError("Training configuration must be a JSON object")
        if args.population is not None:
            n = args.population
            values.update(population_size=n, elite_count=max(1, n // 20),
                          immigrant_count=max(0, n // 20), tournament_size=min(5, n))
        if args.training_seed is not None:
            values["training_seed"] = args.training_seed
        if args.max_steps is not None:
            values["max_steps"] = args.max_steps
        trainer = Trainer(TrainingConfig(**values), output_dir=args.output_dir or "models", initial=initial)
    print(f"TRAINING: {trainer.config.population_size} networks, courses {trainer.config.evaluation_seeds}, "
          f"starting after generation {trainer.generation}", flush=True)
    action = "quit"
    if args.headless:
        trainer.run(args.generations, report=lambda message: print(message, flush=True))
    else:
        import pygame
        from visualization.training_dashboard import TrainingDashboard
        try:
            action = TrainingDashboard(trainer, args.generations).run()
        except pygame.error as exc:
            pygame.quit()
            raise ValueError(f"Unable to open training display: {exc}") from exc
    if trainer.best_genome is not None:
        summary = evaluation_summary(evaluate_network(trainer.champion(), trainer.config.unseen_seeds,
                                                      trainer.config.max_steps, trainer.settings))
        (trainer.output_dir / "evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Champion: {trainer.output_dir / 'best.npz'}; completed generations: {trainer.generation}")
        print("UNSEEN COURSES: " + json.dumps(summary), flush=True)
        if action == "watch":
            from ai.controller import NetworkController
            from game.application import Application
            app = Application(trainer.settings, seed=args.seed, controller=NetworkController(trainer.champion()),
                              auto_restart=not args.no_auto_restart)
            app.start()
            app.run()
            if app.training_requested:
                # Continue the same population rather than discard its learning.
                args.resume = str(trainer.output_dir / "checkpoint.npz")
                args.output_dir = str(trainer.output_dir)
                args.training_config = args.population = args.training_seed = args.max_steps = None
                return run_training(args)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="NeuroFlap — a small, original arcade game")
    parser.add_argument("--seed", type=int, default=None, help="reproducible pipe course")
    parser.add_argument("--headless", action="store_true", help="run a physics smoke check without graphics")
    parser.add_argument("--steps", type=int, default=600, help="maximum headless simulation ticks")
    models = parser.add_mutually_exclusive_group()
    models.add_argument("--create-model", metavar="PATH", help="save a new untrained network and exit (.npz)")
    models.add_argument("--model", metavar="PATH", help="let a saved network control the bird (.npz)")
    parser.add_argument("--network-seed", type=int, default=0, help="initial weights seed for --create-model")
    parser.add_argument("--no-auto-restart", action="store_true", help="keep network watch mode on the game-over screen")
    parser.add_argument("--train", action="store_true", help="evolve network weights (opens a learning view unless --headless)")
    parser.add_argument("--generations", type=int, default=50, help="additional generations to train (default: 50)")
    parser.add_argument("--resume", metavar="CHECKPOINT", help="resume training from a saved population (.npz)")
    parser.add_argument("--training-config", metavar="JSON", help="override genetic algorithm settings from JSON")
    parser.add_argument("--population", type=int, help="population size with 5%% elites and immigrants (default: 100)")
    parser.add_argument("--training-seed", type=int, help="seed for selection, crossover and mutation")
    parser.add_argument("--max-steps", type=int, help="maximum ticks per training/evaluation episode (default: 1800)")
    parser.add_argument("--output-dir", metavar="PATH", help="save champion, checkpoint and statistics here (default: models)")
    parser.add_argument("--evaluate", action="store_true", help="evaluate --model on unseen courses and exit")
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", help="course seeds for --evaluate")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.network_seed < 0:
        parser.error("--network-seed must be nonnegative")
    if args.generations < 1 or (args.max_steps is not None and args.max_steps < 1):
        parser.error("--generations and --max-steps must be positive")
    if args.train and (args.create_model or args.evaluate):
        parser.error("--train cannot be combined with --create-model or --evaluate")
    if args.resume and not args.train:
        parser.error("--resume requires --train")
    if args.evaluate and not args.model:
        parser.error("--evaluate requires --model")
    controller = None
    if args.create_model or args.model:
        try:
            from ai.neural_network import NeuralNetwork
            from ai.controller import NetworkController
            if args.create_model:
                network = NeuralNetwork(seed=args.network_seed)
                network.save(args.create_model)
                print(f"Saved untrained 5 -> 8 -> 1 network ({network.parameter_count} parameters): {args.create_model}")
                return 0
            controller = NetworkController(NeuralNetwork.load(args.model))
        except ModuleNotFoundError as exc:
            if exc.name != "numpy":
                raise
            print("NumPy is missing. Run: python -m pip install -r requirements.txt", file=sys.stderr)
            return 1
        except (OSError, ValueError) as exc:
            print(f"Unable to use the model: {exc}", file=sys.stderr)
            return 1
    if args.train or args.evaluate:
        try:
            if args.train:
                return run_training(args, initial=None if controller is None else controller.network)
            from ai.evaluator import evaluate_network, evaluation_summary
            from ai.genetic_algorithm import TrainingConfig
            seeds = tuple(args.evaluation_seeds) if args.evaluation_seeds else TrainingConfig().unseen_seeds
            print(json.dumps(evaluation_summary(evaluate_network(controller.network, seeds, args.max_steps or 1800))))
            return 0
        except ModuleNotFoundError as exc:
            if exc.name not in ("numpy", "pygame"):
                raise
            print(f"{exc.name} is missing. Run: python -m pip install -r requirements.txt", file=sys.stderr)
            return 1
        except (OSError, ValueError, TypeError) as exc:
            print(f"Unable to train/evaluate: {exc}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            print("Training interrupted; saved completed generations to the checkpoint.", file=sys.stderr)
            return 130
    if args.headless:
        from game.environment import Environment
        env = Environment(seed=args.seed)
        flap_count = 0
        output_min = None
        output_max = None
        for _ in range(args.steps):
            action = controller(env) if controller is not None else False
            flap_count += int(action)
            if controller is not None and controller.last_output is not None:
                output = controller.last_output
                output_min = output if output_min is None else min(output_min, output)
                output_max = output if output_max is None else max(output_max, output)
            env.step(action)
            if env.done:
                break
        print(json.dumps({"seed": args.seed, "ticks": env.ticks, "score": env.score,
                          "death_reason": env.death_reason,
                          "controller": "network" if controller is not None else "no_input",
                          "model": args.model,
                          "flaps": flap_count,
                          "network_output_min": output_min,
                          "network_output_max": output_max,
                          "step_limit_reached": not env.done}))
        return 0
    try:
        import pygame
        from game.application import Application
    except ModuleNotFoundError as exc:
        if exc.name != "pygame":
            raise
        print("Pygame is missing. Run: python -m pip install -r requirements.txt", file=sys.stderr)
        return 1
    try:
        app = Application(seed=args.seed, controller=controller, auto_restart=not args.no_auto_restart)
        app.run()
        if app.training_requested:
            return run_training(args, initial=None if controller is None else controller.network)
    except pygame.error as exc:
        pygame.quit()
        print(f"Unable to open the game: {exc}", file=sys.stderr)
        return 1
    except ModuleNotFoundError as exc:
        if exc.name != "numpy":
            raise
        print("NumPy is missing. Run: python -m pip install -r requirements.txt", file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError) as exc:
        print(f"Unable to train: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Training interrupted; completed generations were saved.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
