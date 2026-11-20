"""Train a bus-route agent and save the results in a timestamped run folder."""

import argparse
import json
import random
from pathlib import Path

from agent import QLearningAgent
from data_loader import load_demand, load_network
from environment import BusRouteEnvironment
from evaluate import evaluate_agent
from q_table_metrics import print_q_table_summary, summarize_q_table
from run_logging import RunLogger
from train import train_agent
from visualize import plot_route_network, plot_training_rewards


RUNS_DIR = Path(__file__).resolve().parent / "runs"


def parse_arguments():
    """Read optional command-line settings and check their values."""
    parser = argparse.ArgumentParser(
        description="Train and record a bus-route Q-learning run."
    )
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also display saved plots",
    )
    args = parser.parse_args()

    if args.episodes < 1 or args.eval_interval < 1:
        parser.error("episodes and eval-interval must be positive")

    return args


def build_config(args):
    """Keep all recorded training and model settings together."""
    return {
        "algorithm": "Q-learning",
        "seed": args.seed,
        "episodes": args.episodes,
        "evaluation_seed": 12345,
        "eval_interval": args.eval_interval,
        "environment": {
            "max_routes": 4,
            "max_stops_per_route": 8,
            "cost_weight": 0.0,
        },
        "agent": {
            "alpha": 0.5,
            "gamma": 1.0,
            "epsilon": 1.0,
            "epsilon_min": 0.05,
            "epsilon_decay_episodes": max(1, (4 * args.episodes + 4) // 5),
        },
    }


def evaluate_with_fixed_seed(env, agent, seed):
    """Use repeatable evaluation without advancing training's random sequence."""
    training_random_state = random.getstate()
    try:
        random.seed(seed)
        return evaluate_agent(env, agent)
    finally:
        random.setstate(training_random_state)


def train_model(network, demand, config, run):
    """Train the agent and record evaluations at the configured interval."""
    training_env = BusRouteEnvironment(network, demand, **config["environment"])
    evaluation_env = BusRouteEnvironment(network, demand, **config["environment"])
    agent = QLearningAgent(**config["agent"])
    final_results = None

    def record_evaluation(episode):
        nonlocal final_results
        final_results = evaluate_with_fixed_seed(
            evaluation_env, agent, config["evaluation_seed"]
        )
        run.record_evaluation(episode, final_results)
        print(
            f"Evaluation {episode}: "
            f"coverage={final_results['coverage']:.2f}% "
            f"objective={final_results['objective']:.2f}"
        )

    print("Training Q-learning agent...")
    rewards = train_agent(
        training_env,
        agent,
        episodes=config["episodes"],
        on_episode=run.record_episode,
        on_evaluate=record_evaluation,
        eval_interval=config["eval_interval"],
    )
    return agent, rewards, final_results


def print_results(results):
    """Print JSON with each route and decision on a single line."""
    print("{")
    items = list(results.items())

    for index, (key, value) in enumerate(items):
        comma = "," if index < len(items) - 1 else ""
        label = json.dumps(key)

        if key in ("routes", "decisions") and value:
            print(f"  {label}: [")
            for row_index, row in enumerate(value):
                row_comma = "," if row_index < len(value) - 1 else ""
                print(f"    {json.dumps(row)}{row_comma}")
            print(f"  ]{comma}")
        else:
            print(f"  {label}: {json.dumps(value)}{comma}")

    print("}")


def save_results(network, demand, config, agent, rewards, results, run, show_plots):
    """Save final routes and figures, then print the result summary."""
    run.save_q_table(agent)
    run.q_table_metrics = summarize_q_table(
        agent, network, demand, config["environment"],
    )
    run.q_table_metrics["csv_size_bytes"] = (run.path / "q_table.csv").stat().st_size
    run.write_json("final_results.json", results)
    plot_route_network(
        network,
        results,
        output_path=run.path / "final_routes.png",
        show=show_plots,
    )
    plot_training_rewards(
        rewards,
        output_path=run.path / "training_rewards.png",
        show=show_plots,
    )
    print_results(results)
    print_q_table_summary(run.q_table_metrics)


def main():
    args = parse_arguments()
    config = build_config(args)
    random.seed(config["seed"])

    run = RunLogger(RUNS_DIR, config)
    print("Run folder:", run.path)

    try:
        network = load_network()
        demand = load_demand()
        agent, rewards, results = train_model(network, demand, config, run)
        save_results(
            network, demand, config, agent, rewards, results, run,
            show_plots=args.show,
        )
        run.finish("completed")
    except BaseException as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        run.finish(status, f"{type(error).__name__}: {error}")
        raise
    finally:
        run.close()


if __name__ == "__main__":
    main()
