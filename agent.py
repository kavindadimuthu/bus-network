import random
from collections import defaultdict


class QLearningAgent:
    def __init__(
        self,
        alpha=0.5,
        gamma=1.0,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay_episodes=10000
    ):
        if not 0 <= epsilon_min <= epsilon <= 1:
            raise ValueError("Require 0 <= epsilon_min <= epsilon <= 1")
        if not isinstance(epsilon_decay_episodes, int) or epsilon_decay_episodes < 1:
            raise ValueError("epsilon_decay_episodes must be a positive integer")
        self.q_table = defaultdict(dict)
        self.updated_state_action_pairs = 0

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_start = epsilon
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.epsilon_decay_steps = 0

    def q_value(self, state, action):
        return self.q_table[state].get(action, 0.0)

    def choose_action(self, state, actions, explore=True):
        if explore and random.random() < self.epsilon:
            return random.choice(actions)

        best_q = max(
            self.q_value(state, action)
            for action in actions
        )

        best_actions = [
            action
            for action in actions
            if self.q_value(state, action) == best_q
        ]

        return random.choice(best_actions)

    def update(
        self,
        state,
        action,
        reward,
        next_state,
        next_actions,
        done
    ):
        old_q = self.q_value(state, action)

        if done:
            target = reward
        else:
            best_next_q = max(
                self.q_value(next_state, a)
                for a in next_actions
            )
            target = reward + self.gamma * best_next_q

        if action not in self.q_table[state]:
            self.updated_state_action_pairs += 1

        new_q = old_q + self.alpha * (target - old_q)
        self.q_table[state][action] = new_q
        return abs(new_q - old_q)

    def decay_epsilon(self):
        """Decrease epsilon linearly after each episode, then hold the floor."""
        self.epsilon_decay_steps += 1
        if self.epsilon_decay_steps >= self.epsilon_decay_episodes:
            self.epsilon = self.epsilon_min
            return
        progress = min(self.epsilon_decay_steps / self.epsilon_decay_episodes, 1.0)
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon_start - (self.epsilon_start - self.epsilon_min) * progress
        )
