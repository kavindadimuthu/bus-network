def train_agent(env, agent, episodes=30000, on_episode=None, on_evaluate=None, eval_interval=1000):
    rewards_history = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        total_abs_q_change = 0.0

        while not env.done:
            actions = env.available_actions()

            action = agent.choose_action(
                state, actions, explore=True
            )

            next_state, reward, done = env.step(action)
            next_actions = env.available_actions()

            total_abs_q_change += agent.update(
                state,
                action,
                reward,
                next_state,
                next_actions,
                done
            )

            steps += 1
            total_reward += reward
            state = next_state

        rewards_history.append(total_reward)
        agent.decay_epsilon()
        if on_episode:
            on_episode(
                episode + 1, total_reward, steps, agent,
                mean_abs_q_change=total_abs_q_change / steps if steps else 0.0,
            )
        if on_evaluate and ((episode + 1) % eval_interval == 0 or episode + 1 == episodes):
            on_evaluate(episode + 1)

        if (episode + 1) % 5000 == 0:
            recent = rewards_history[-5000:]

            print(
                f"Episode {episode + 1} | "
                f"Average score: {sum(recent)/len(recent):.2f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    return rewards_history