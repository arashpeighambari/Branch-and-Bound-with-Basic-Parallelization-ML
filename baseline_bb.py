import math


nodes_expanded_std = 0


# Compute MST weight over a set of nodes using Prim's algorithm
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


# Standard branch-and-bound using MST and minimum edge bounds
def branch_and_bound_standard(current, visited, path, cost_so_far, cost, n, best):
    global nodes_expanded_std
    nodes_expanded_std += 1

    start = 0
    if len(visited) == n:
        total_cost = cost_so_far + cost[current][start]
        if total_cost < best[0]:
            best[0] = total_cost
            best[1] = path + [start]
        return

    remaining = set(range(n)) - visited
    if remaining:
        mst = mst_weight(remaining, cost)
        min_from_current = min([cost[current][r] for r in remaining])
        min_to_start = min([cost[r][start] for r in remaining])
        lower_bound = cost_so_far + mst + min_from_current + min_to_start
    else:
        lower_bound = cost_so_far + cost[current][start]

    if lower_bound >= best[0]:
        return

    for nxt in remaining:
        branch_and_bound_standard(
            nxt,
            visited | {nxt},
            path + [nxt],
            cost_so_far + cost[current][nxt],
            cost,
            n,
            best
        )
