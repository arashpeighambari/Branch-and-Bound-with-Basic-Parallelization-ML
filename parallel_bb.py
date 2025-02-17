import multiprocessing
from baseline_bb import mst_weight 

def branch_and_bound_parallel(current, visited, path, cost_so_far, cost, n, best, lock):

    # A shared dictionary best is added and a lock to prune branches.
    
    start = 0
    if len(visited) == n:
        total_cost = cost_so_far + cost[current][start]
        with lock:
            if total_cost < best['cost']:
                best['cost'] = total_cost
                best['path'] = path + [start]
        return

    remaining = set(range(n)) - visited
    if remaining:
        mst = mst_weight(remaining, cost)
        lower_bound = cost_so_far + mst
    else:
        lower_bound = cost_so_far + cost[current][start]
    
    with lock:
        current_best = best['cost']
    if lower_bound >= current_best:
        return

    for nxt in remaining:
        branch_and_bound_parallel(
            nxt,
            visited | {nxt},
            path + [nxt],
            cost_so_far + cost[current][nxt],
            cost,
            n,
            best,
            lock
        )

def worker(args):

    (initial_city, cost, n, best, lock) = args
    # Start the tour from city 0
    branch_and_bound_parallel(
        initial_city,
        {0, initial_city},
        [0, initial_city],
        cost[0][initial_city],
        cost,
        n,
        best,
        lock
    )
    
    return