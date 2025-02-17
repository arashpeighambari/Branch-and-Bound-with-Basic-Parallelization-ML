import math
import random
import itertools
import numpy as np
from sklearn.linear_model import LinearRegression

nodes_expanded_ml = 0


def mst_weight(nodes, cost):
    if not nodes:
        return 0
    nodes = list(nodes)
    used = {nodes[0]}
    total = 0
    while len(used) < len(nodes):
        best_edge = math.inf
        best_node = None
        for u in used:
            for v in nodes:
                if v not in used and cost[u][v] < best_edge:
                    best_edge = cost[u][v]
                    best_node = v
        total += best_edge
        used.add(best_node)
    return total


#  Brute-force optimal additional cost that is used to label the targets in generate_training_data()
def optimal_completion_cost(visited, current, cost):
    n = len(cost)
    remaining = list(set(range(n)) - visited)
    if not remaining:
        return cost[current][0]
    best_additional = math.inf
    for perm in itertools.permutations(remaining):
        additional = cost[current][perm[0]]
        for i in range(len(perm) - 1):
            additional += cost[perm[i]][perm[i+1]]
        additional += cost[perm[-1]][0]
        best_additional = min(best_additional, additional)
    return best_additional


#  average edge cost among for feature 5 (f5).
def average_edge_cost(nodes, cost):
    if len(nodes) < 2:
        return 0
    total = 0
    count = 0
    nodes = list(nodes)
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            total += cost[nodes[i]][nodes[j]]
            count += 1
    return total / count if count > 0 else 0


# Generate training data for the ML model
def generate_training_data(cost, n, num_samples=1000):
    X = []
    y = []
    for _ in range(num_samples):
        route_length = random.randint(1, n - 1)
        nodes_list = list(range(1, n))
        random.shuffle(nodes_list)
        partial = [0] + nodes_list[:route_length]
        visited = set(partial)
        current = partial[-1]
        remaining = set(range(n)) - visited

        f1 = mst_weight(remaining, cost) if remaining else 0
        f2 = min([cost[current][r] for r in remaining]) if remaining else cost[current][0]
        f3 = min([cost[r][0] for r in remaining]) if remaining else cost[current][0]
        f4 = len(remaining)
        f5 = average_edge_cost(remaining, cost) if remaining else 0

        features = [f1, f2, f3, f4, f5]
        target = optimal_completion_cost(visited, current, cost)
        X.append(features)
        y.append(target)
    return np.array(X), np.array(y)


# Compute features for a given partial tour.
def compute_features(visited, current, cost, n):
    remaining = set(range(n)) - visited
    f1 = mst_weight(remaining, cost) if remaining else 0
    f2 = min([cost[current][r] for r in remaining]) if remaining else cost[current][0]
    f3 = min([cost[r][0] for r in remaining]) if remaining else cost[current][0]
    f4 = len(remaining)
    f5 = average_edge_cost(remaining, cost) if remaining else 0
    return [f1, f2, f3, f4, f5]


# Train the ML model.
def train_ml_model(cost, n, num_samples=1000):
    print("Generating training data for ML heuristic...")
    X_train, y_train = generate_training_data(cost, n, num_samples)
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("ML training complete.")
    print("Model coefficients:", model.coef_)
    print("Model intercept:", model.intercept_)
    return model


# Branch-and-bound using the ML-augmented approach
def branch_and_bound_ml(current, visited, path, cost_so_far, cost, n, best, ml_model):
    global nodes_expanded_ml
    nodes_expanded_ml += 1

    start = 0
    if len(visited) == n:
        total_cost = cost_so_far + cost[current][start]
        if total_cost < best[0]:
            best[0] = total_cost
            best[1] = path + [start]
        return

    features = compute_features(visited, current, cost, n)
    predicted_additional = ml_model.predict(np.array(features).reshape(1, -1))[0]
    lower_bound = cost_so_far + predicted_additional

    if lower_bound >= best[0]:
        return

    remaining = set(range(n)) - visited
    for nxt in remaining:
        branch_and_bound_ml(
            nxt,
            visited | {nxt},
            path + [nxt],
            cost_so_far + cost[current][nxt],
            cost,
            n,
            best,
            ml_model
        )
