class BusRouteEnvironment:
    def __init__(
        self,
        network,
        demand,
        max_routes=3,
        max_stops_per_route=8,
        cost_weight=2.0
    ):
        self.network = network
        self.demand = demand

        self.max_routes = max_routes
        self.max_stops_per_route = max_stops_per_route
        self.cost_weight = cost_weight

        self.reset()

    def reset(self):
        self.completed_routes = ()
        self.current_route = ()
        self.done = False

        return self.get_state()

    def get_state(self):
        return (
            self.completed_routes,
            self.current_route
        )

    def route_time(self, route):
        return sum(
            self.network[a][b]
            for a, b in zip(route, route[1:])
        )

    def served_demand(self, routes):
        total = 0

        for (origin, destination), passengers in self.demand.items():
            if any(
                origin in route and destination in route
                for route in routes
            ):
                total += passengers

        return total

    def objective(self, routes):
        served = self.served_demand(routes)

        route_cost = sum(
            self.route_time(route)
            for route in routes
        )

        return served - self.cost_weight * route_cost

    def available_actions(self):
        if self.done:
            return []

        if not self.current_route:
            actions = [("STOP",)]

            if len(self.completed_routes) < self.max_routes:
                for stop in sorted(self.network):
                    actions.append(("START", stop))

            return actions

        last_stop = self.current_route[-1]
        actions = []

        if len(self.current_route) < self.max_stops_per_route:
            for neighbor in sorted(self.network[last_stop]):
                if neighbor not in self.current_route:
                    actions.append(("EXTEND", neighbor))

        if len(self.current_route) >= 2:
            actions.append(("FINISH_ROUTE",))

        return actions

    def step(self, action):
        if action not in self.available_actions():
            raise ValueError(f"Invalid action: {action}")

        old_score = self.objective(self.completed_routes)

        action_type = action[0]

        if action_type == "START":
            self.current_route = (action[1],)

        elif action_type == "EXTEND":
            self.current_route += (action[1],)

        elif action_type == "FINISH_ROUTE":
            self.completed_routes += (self.current_route,)
            self.current_route = ()

        elif action_type == "STOP":
            self.done = True

        new_score = self.objective(self.completed_routes)
        reward = new_score - old_score

        return self.get_state(), reward, self.done