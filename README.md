# Training run records

Run `python main.py`. Each invocation creates its own timestamped folder under
`runs/`, with an explicit local UTC offset in the metadata. Generated runs are
ignored by Git. No tracking service or additional package is required.

For a quick experiment: `python main.py --episodes 100 --seed 42 --eval-interval 50`.
Defaults: 10,000 episodes, seed 42, evaluation every 1,000 episodes and at the end.
Plots are saved automatically; add `--show` to display them too.

Exploration decreases linearly from epsilon 1.0 to 0.05 over the first 80% of
training episodes (rounded up), then stays at 0.05 for the remaining episodes.
Epsilon is updated after each episode; the first episode uses 1.0. The run config
records `epsilon_decay_episodes` instead of the former exponential multiplier.
For 10,000 episodes the decrement is 0.00011875 per episode, giving epsilon
0.525 after 4,000 episodes and 0.05 after 8,000 episodes. The final 2,000
episodes use 0.05. Standalone agents default to a 10,000-episode decay horizon;
`main.py` sets it to 80% of the requested `--episodes` value, with a minimum of one.

Each folder contains:

- `config.json`: training, agent, environment and evaluation settings.
- `metadata.json`: start time, Git revision/dirty flag and input CSV hashes.
- `training_metrics.csv`: episode reward, steps, epsilon after decay, Q-table size
  and elapsed seconds. Rows are flushed after every completed episode.
- `evaluation_metrics.csv`: served/total demand, coverage (%), route time (minutes)
  and objective. Evaluations use a separate environment and fixed tie-breaking
  seed without advancing training's random stream.
- `q_table.csv`: the final Q-table, exported once after training for inspection.
- `best_results.json` and `final_results.json`: evaluated routes and metrics.
- `final_routes.png` and `training_rewards.png`: saved figures.
- `summary.json`: running/completed/interrupted/failed status, timing, completed
  episodes, best objective and error if any. A hard process kill may leave running.

An interruption preserves completed metric rows. The final Q-table CSV is saved
only when training finishes.
The final plot describes the final agent; best_results.json describes the best one.
This environment measures demand coverage with at most one transfer and route cost.
Routes operate in both directions. An OD pair is served when both stops share a
route, or when an origin route and a destination route share a transfer stop.
Each OD pair counts once; journeys needing two or more transfers are unserved.
Rewards use the change in the objective after each action, including the route
currently being built once it has two stops. Finishing a route does not reward
its demand again. Passenger waiting times, capacity, and transfer timing are not
simulated. Older runs used direct-only coverage and are not directly comparable.
Evaluation currently uses
the same static network/demand as training, so it does not measure generalization.

## Inspecting the Q-table

Open `runs/YOUR_RUN/q_table.csv` in Excel or another CSV viewer. Each stored
state/action pair has its own row with these columns:

| Column | Meaning |
| --- | --- |
| `state_id` | Identifier within this export; repeated for actions in the same state |
| `completed_routes` | Completed routes, written as a JSON list of lists |
| `current_route` | Route being built, written as a JSON list |
| `action` | START, EXTEND, FINISH_ROUTE, or STOP |
| `next_stop` | Stop number for START/EXTEND; blank for other actions |
| `q_value` | Actual stored Q-value, without display rounding |

States with no stored action values have one row with blank action, next_stop and
q_value cells. A blank value is not a learned zero. Unstored actions are not
expanded into CSV rows; the agent treats them as 0.0 when choosing actions.
State IDs are local to each export, so compare full states across runs.
The CSV contains the final agent's table, which may differ from the agent that
produced `best_results.json`. No model checkpoints are written.

## Q-table diagnostics

`summary.json` includes `q_table_metrics`: encountered states, stored state-action
pairs, states without stored actions, nonzero stored Q-values, minimum/maximum/mean
stored Q-values, legal action count across encountered states, action coverage,
and Q-table CSV size in bytes. Empty tables have null Q-value statistics and null
coverage when no legal actions exist. Statistics include stored zero values.

Coverage is stored pairs divided by legal actions across Q-table states, not a
percentage of the entire theoretical state space. Encountered states include
states inspected during training and evaluation; some have no updates.

`training_metrics.csv` additionally records cumulative `updated_state_action_pairs`
and `mean_abs_q_change` (mean absolute change per update in that episode, including
zero changes). Smaller changes alone do not prove convergence or policy quality.
The terminal prints only states, updated pairs, encountered-state action coverage,
and Q-value range at the end. Existing route/result output remains available.
