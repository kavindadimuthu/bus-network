"""Small, file-based experiment records; no tracking service required."""
import csv
import hashlib
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path


class RunLogger:
    def __init__(self, root, config):
        self.started = datetime.now().astimezone()
        self.clock = time.perf_counter()
        self.path = Path(root) / (self.started.strftime("%Y-%m-%d_%H-%M-%S_%f")
                                  + f"_qlearning_seed{config['seed']}")
        self.path.mkdir(parents=True)
        self.completed_episodes = 0
        self.best_objective = None
        self.files = []
        self.q_table_metrics = None
        self.write_json("config.json", config)
        project = Path(__file__).resolve().parent
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=project, text=True,
                stderr=subprocess.DEVNULL).strip()
            dirty = bool(subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=project, text=True))
        except (OSError, subprocess.CalledProcessError):
            commit, dirty = None, None
        self.write_json("metadata.json", {
            "started_at": self.started.isoformat(), "git_commit": commit,
            "git_dirty": dirty,
            "data_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                            for f in sorted((project / "data").glob("*.csv"))},
        })
        self.training = self.csv("training_metrics.csv", [
            "episode", "reward", "steps", "epsilon", "q_table_states", "updated_state_action_pairs",
            "mean_abs_q_change", "elapsed_seconds"])
        self.evaluation = self.csv("evaluation_metrics.csv", [
            "episode", "served_demand", "total_demand", "coverage", "route_time", "objective"])
        self.finish("running")

    def write_json(self, name, value):
        (self.path / name).write_text(json.dumps(value, indent=2), encoding="utf-8")

    def csv(self, name, fields):
        file = (self.path / name).open("w", newline="", encoding="utf-8")
        self.files.append(file)
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        file.flush()
        return writer, file

    def record_episode(self, episode, reward, steps, agent, mean_abs_q_change=0.0):
        self.completed_episodes = episode
        writer, file = self.training
        writer.writerow(dict(episode=episode, reward=reward, steps=steps,
                             epsilon=agent.epsilon, q_table_states=len(agent.q_table),
                             updated_state_action_pairs=agent.updated_state_action_pairs,
                             mean_abs_q_change=mean_abs_q_change,
                             elapsed_seconds=round(time.perf_counter() - self.clock, 3)))
        file.flush()

    def save_q_table(self, agent):
        """Export stored Q-values, one state/action pair per row, for inspection."""
        fields = [
            "state_id", "completed_routes", "current_route",
            "action", "next_stop", "q_value",
        ]
        with (self.path / "q_table.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()

            for state_id, (state, actions) in enumerate(agent.q_table.items(), 1):
                completed_routes, current_route = state
                state_columns = {
                    "state_id": state_id,
                    "completed_routes": json.dumps(completed_routes),
                    "current_route": json.dumps(current_route),
                }
                if not actions:
                    # Keep visited states with no stored action values visible.
                    writer.writerow(state_columns)
                    continue

                for action, q_value in actions.items():
                    writer.writerow({
                        **state_columns,
                        "action": action[0],
                        "next_stop": action[1] if len(action) > 1 else "",
                        "q_value": q_value,
                    })

    def record_evaluation(self, episode, results):
        writer, file = self.evaluation
        writer.writerow({"episode": episode, **{
            key: results[key] for key in writer.fieldnames if key != "episode"}})
        file.flush()
        if self.best_objective is None or results["objective"] > self.best_objective:
            self.best_objective = results["objective"]
            self.write_json("best_results.json", results)

    def finish(self, status, error=None):
        self.write_json("summary.json", {
            "status": status, "started_at": self.started.isoformat(),
            "ended_at": None if status == "running" else datetime.now().astimezone().isoformat(),
            "elapsed_seconds": round(time.perf_counter() - self.clock, 3),
            "completed_episodes": self.completed_episodes,
            "best_evaluation_objective": self.best_objective, "error": error,
            "q_table_metrics": self.q_table_metrics,
        })

    def close(self):
        for file in self.files:
            file.close()
