def train_agent(env, agent, episodes=30000):
    rewards_history = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        while not env.done:
            actions = env.available_actions()

            action = agent.choose_action(
                state, actions, explore=True
            )

            next_state, reward, done = env.step(action)
            next_actions = env.available_actions()

            agent.update(
                state,
                action,
                reward,
                next_state,
                next_actions,
                done
            )

            total_reward += reward
            state = next_state

        rewards_history.append(total_reward)
        agent.decay_epsilon()

        if (episode + 1) % 5000 == 0:
            recent = rewards_history[-5000:]

            print(
                f"Episode {episode + 1} | "
                f"Average score: {sum(recent)/len(recent):.2f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    return rewards_history