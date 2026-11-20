import csv
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_network(filename="mandl_links(1).csv"):
    network = {}

    with open(DATA_DIR / filename, newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            source = int(row["source"])
            target = int(row["target"])
            time = float(row["travel_time_min"])

            if time <= 0:
                raise ValueError("Travel times must be positive")

            network.setdefault(source, {})
            network.setdefault(target, {})

            # Assume two-way road connections
            network[source][target] = time
            network[target][source] = time

    return network


def load_demand(filename="mandl_od_matrix(1).csv"):
    demand = {}

    with open(DATA_DIR / filename, newline="") as file:
        reader = csv.DictReader(file)

        destinations = [
            int(name)
            for name in reader.fieldnames[1:]
        ]

        for row in reader:
            origin = int(row["origin/destination"])

            for destination in destinations:
                passengers = float(row[str(destination)])

                if passengers < 0:
                    raise ValueError("Demand cannot be negative")

                if origin != destination and passengers > 0:
                    demand[(origin, destination)] = passengers

    return demand