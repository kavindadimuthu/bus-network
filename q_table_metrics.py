"""Final diagnostics for stored Q-values and actions in encountered states."""
from environment import BusRouteEnvironment


def summarize_q_table(agent, network, demand, environment_config):
    """Count legal actions in table states, without altering the training environment."""
    inspection_env = BusRouteEnvironment(network, demand, **environment_config)
    pairs = 0
    empty_states = 0
    nonzero = 0
    total = 0.0
    minimum = None
    maximum = None
    legal_pairs = 0

    for state, actions in agent.q_table.items():
        inspection_env.completed_routes, inspection_env.current_route = state
        # Q-table keys describe decision states; STOP's terminal state is not added.
        inspection_env.done = False
        legal_pairs += len(inspection_env.available_actions())
        empty_states += not actions
        for value in actions.values():
            pairs += 1
            nonzero += value != 0.0
            total += value
            minimum = value if minimum is None else min(minimum, value)
            maximum = value if maximum is None else max(maximum, value)

    return {
        "encountered_states": len(agent.q_table),
        "stored_state_action_pairs": pairs,
        "states_without_stored_actions": empty_states,
        "nonzero_q_values": nonzero,
        "min_q_value": minimum,
        "max_q_value": maximum,
        "mean_q_value": total / pairs if pairs else None,
        "legal_state_action_pairs_in_encountered_states": legal_pairs,
        "encountered_state_action_coverage_percent": (
            100 * pairs / legal_pairs if legal_pairs else None
        ),
    }


def print_q_table_summary(metrics):
    """Keep terminal diagnostics short; full details are saved in summary.json."""
    print("\nQ-table summary:")
    print(f"  Encountered states: {metrics['encountered_states']:,}")
    print(f"  Updated state-action pairs: {metrics['stored_state_action_pairs']:,}")
    coverage = metrics["encountered_state_action_coverage_percent"]
    if coverage is not None:
        print(f"  Action coverage in encountered states: {coverage:.2f}%")
    if metrics["min_q_value"] is not None:
        print(f"  Q-value range: {metrics['min_q_value']:.3f} to "
              f"{metrics['max_q_value']:.3f}")
