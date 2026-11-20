import matplotlib.pyplot as plt

import numpy as np

def plot_training_rewards(rewards_history, window=50):

    plt.figure(figsize=(10, 5))

    # Raw episode rewards

    plt.plot(

        rewards_history,

        alpha=0.3,

        label="Episode Reward"

    )

    # Moving average

    if len(rewards_history) >= window:

        moving_average = np.convolve(

            rewards_history,

            np.ones(window) / window,

            mode="valid"

        )

        plt.plot(

            range(window - 1, len(rewards_history)),

            moving_average,

            label=f"Moving Average ({window})"

        )

    plt.xlabel("Episode")

    plt.ylabel("Total Reward")

    plt.title("Bus Route Q-Learning Training")

    plt.legend()

    plt.grid(True)

    plt.show()

def plot_route_network(network, results, output_path=None, show=True):
    """Draw every road and overlay evaluated routes on a fixed reference layout."""
    from pathlib import Path
    from matplotlib.lines import Line2D

    stops = sorted(network)
    index = {stop: i for i, stop in enumerate(stops)}
    # Fixed schematic coordinates traced from the supplied Mandl reference.
    # These are drawing coordinates, not latitude/longitude.
    reference = {
        1: (62, -18), 2: (104, -57), 3: (166, -57),
        4: (103, -117), 5: (55, -147), 6: (166, -117),
        7: (299, -191), 8: (215, -161), 9: (368, -86),
        10: (224, -224), 11: (166, -276), 12: (103, -230),
        13: (237, -305), 14: (280, -276), 15: (296, -129),
    }
    missing = set(stops) - reference.keys()
    if missing:
        raise ValueError(f"Missing reference positions for stops: {sorted(missing)}")
    positions = np.array([reference[stop] for stop in stops], dtype=float) / 300
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.subplots_adjust(right=0.73)
    for source in stops:
        for target, minutes in network[source].items():
            if index[source] >= index[target]:
                continue
            points = positions[[index[source], index[target]]]
            ax.plot(*points.T, color='#cbd5e1', linewidth=2, zorder=1)
            middle = points.mean(axis=0)
            ax.text(*middle, f'{minutes:g}', fontsize=8, color='#64748b',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    routes = results['routes']
    colors = plt.get_cmap('tab10')
    handles = []
    for i, route in enumerate(routes):
        color = colors(i % 10)
        # Offset shared segments so each route remains visible.
        for source, target in zip(route, route[1:]):
            points = positions[[index[source], index[target]]].copy()
            delta = points[1] - points[0]
            length = np.linalg.norm(delta)
            if length:
                normal = np.array([-delta[1], delta[0]]) / length
                if index[source] > index[target]:
                    normal = -normal
                points += normal * (i - (len(routes) - 1) / 2) * 0.018
            ax.plot(*points.T, color=color, linewidth=3, alpha=0.85, zorder=2)
        handles.append(Line2D([0], [0], color=color, lw=3,
                              label=f"Route {i + 1}: " + ' -> '.join(map(str, route))))
    served_stops = {stop for route in routes for stop in route}
    ax.scatter(*positions.T, s=420,
               c=['#dbeafe' if stop in served_stops else '#f1f5f9' for stop in stops],
               edgecolors='#334155', zorder=3)
    for stop, point in zip(stops, positions):
        ax.text(*point, str(stop), ha='center', va='center', weight='bold', zorder=4)
    ax.set_title('Final bus route design - full network', fontsize=18, pad=20)
    ax.set_aspect('equal')
    ax.margins(0.15)
    ax.axis('off')
    if handles:
        fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.73, 0.88), fontsize=9)
    summary = (f"Demand coverage: {results['coverage']:.2f}%\n"
               f"Served demand: {results['served_demand']:g} / {results['total_demand']:g}\n"
               f"Total route time: {results['route_time']:g} min\n"
               f"Objective: {results['objective']:g}\n\n"
               'Gray lines: all road links\nEdge labels: travel minutes\n'
               'Blue stops: included in routes\n\nFixed Mandl reference layout\nSchematic; not geographic coordinates')
    if not routes:
        summary = 'No routes selected by the agent.\n\n' + summary
    fig.text(0.74, 0.45, summary, va='top', fontsize=10, linespacing=1.7)
    if output_path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=180, bbox_inches='tight')
    if show:
        plt.show()
    return fig


if __name__ == '__main__':
    import json
    from pathlib import Path
    from data_loader import load_network

    output_dir = Path(__file__).resolve().parent / 'outputs'
    results_path = output_dir / 'final_results.json'
    if not results_path.exists():
        raise SystemExit('Run main.py first to generate final_results.json.')
    results = json.loads(results_path.read_text())
    plot_route_network(load_network(), results, output_dir / 'final_routes.png')
