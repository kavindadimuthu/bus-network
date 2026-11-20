def evaluate_agent(env, agent):
    state = env.reset()
    decisions = []

    while not env.done:
        actions = env.available_actions()

        action = agent.choose_action(
            state,
            actions,
            explore=False
        )

        decisions.append(action)

        state, reward, done = env.step(action)

    routes = env.completed_routes

    served = env.served_demand(routes)
    total_demand = sum(env.demand.values())

    return {
        "routes": routes,
        "decisions": decisions,
        "served_demand": served,
        "total_demand": total_demand,
        "coverage": 100 * served / total_demand,
        "route_time": sum(env.route_time(r) for r in routes),
        "objective": env.objective(routes),
    }