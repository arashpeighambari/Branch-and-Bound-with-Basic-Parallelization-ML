import math
import itertools

def brute_force_tsp(cost):

    #checking all permutations.
    n = len(cost)
    best_cost = math.inf
    best_path = []
    
    for perm in itertools.permutations(range(1, n)):
        tour = [0] + list(perm) + [0]
        tour_cost = sum(cost[tour[i]][tour[i+1]] for i in range(len(tour)-1))
        if tour_cost < best_cost:
            best_cost = tour_cost
            best_path = tour
    return best_cost, best_path