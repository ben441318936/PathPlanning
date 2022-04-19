import numpy as np
from pqdict import pqdict

class Weighted_A_star(object):
    def __init__(self) -> None:
        pass

    def weighted_A_star(self, map: np.ndarray, start, goal, neighbors_func, eps=1) -> bool:
        '''
        Assume that the current map is correct, plan a path to the target.
        Using weighted A* with Euclidean distance as heuristic.
        This heuristic is consistent for all k-connected grid.
        '''
        # Initialize the data structures
        # Labels
        g = np.ones(map.shape) * np.inf
        start_ind = tuple(start)
        goal_ind = tuple(goal)
        g[start_ind] = 0
        # Priority queue for OPEN list
        OPEN = pqdict({})
        OPEN[start_ind] = g[start_ind] + eps * np.linalg.norm(start - goal)
        # Predecessor matrix to keep track of path
        # create dtype string
        dtype_string = ",".join(['i' for _ in range(len(map.shape))])
        pred = np.full(map.shape, -1, dtype=dtype_string)

        done = False

        while not done:
            if len(OPEN) != 0 :
                parent_ind = OPEN.popitem()[0]
            else:
                break
            if parent_ind == goal_ind:
                done = True
                break
            # get neighbors
            children_inds, children_costs = neighbors_func(parent_ind)
            # # Get list of children
            # children_inds = self._get_neighbors_inds(parent_ind)
            # # Get list of costs from parent to children
            # children_costs = self._get_parent_children_costs(parent_ind)
            for child_ind, child_cost in zip(children_inds, children_costs):
                if child_ind is not None:
                    if g[child_ind] > g[parent_ind] + children_costs[j]:
                        g[child_ind] = g[parent_ind] + children_costs[j]
                        pred[child_ind] = parent_ind
                        # This updates if child already in OPEN
                        # and appends to OPEN otherwise
                        OPEN[child_ind] = g[child_ind] + eps * np.linalg.norm(np.array(child_ind) - np.array(goal_ind))

        # We have found a path
        path = []
        if done:
            ind = goal_ind
            while True:
                path.append(ind)
                if ind == start_ind:
                    break
                else:
                    ind = pred[ind]
        path = list(reversed(path))
        path = np.array(path)
        return done, path