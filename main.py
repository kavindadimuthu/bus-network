import random
import json
from pathlib import Path

from data_loader import load_network, load_demand
from environment import BusRouteEnvironment
from agent import QLearningAgent
from train import train_agent
from evaluate import evaluate_agent
from visualize import plot_training_rewards, plot_route_network


def main():
    # random.seed(42)

    network = load_network()
    demand = load_demand()

    print("Mandl Network Loaded")
    print("--------------------")
    print("Stops:", len(network))
    print("OD demand:", sum(demand.values()))

    env = BusRouteEnvironment(
        network=network,
        demand=demand,
        max_routes=4,
        max_stops_per_route=8,
        cost_weight=0.0
    )

    agent = QLearningAgent(
        alpha=0.5,
        gamma=0.9,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.9995
    )

    print("\nTraining Q-Learning agent...")

    rewards = train_agent(
        env,
        agent,
        episodes=10000
    )

    results = evaluate_agent(env, agent)

    print("\nDESIGNED ROUTE NETWORK")
    print("----------------------")

    for i, route in enumerate(results["routes"], 1):
        print(
            f"Route {i}: "
            + " -> ".join(map(str, route))
        )

    print("\nPerformance:")
    print("Served demand:", results["served_demand"])
    print("Total demand:", results["total_demand"])
    print("Coverage:", results["coverage"], "%")
    print("Route time:", results["route_time"])
    print("Objective:", results["objective"])

    print("\nQ-table states:", len(agent.q_table))

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "final_results.json").write_text(json.dumps(results, indent=2))
    plot_route_network(network, results, output_dir / "final_routes.png", show=False)
    print("Final route graph saved to:", output_dir / "final_routes.png")
    plot_training_rewards(rewards)


if __name__ == "__main__":
    main()