import numpy as np
from pqdict import pqdict

class Weighted_A_star(object):
    def __init__(self, eps=1) -> None:
        self.eps = eps

    def plan(self, map: np.ndarray, start, goal, neighbors_func) -> tuple:
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
        OPEN[start_ind] = g[start_ind] + self.eps * np.linalg.norm(start - goal)
        # Predecessor matrix to keep track of path
        # # create dtype string
        # dtype_string = ",".join(['i' for _ in range(len(map.shape))])
        # pred = np.full(map.shape, -1, dtype=dtype_string)
        pred = np.full((map.shape[0], map.shape[1], 2), -1, dtype=int)

        done = False

        while not done:
            if len(OPEN) != 0 :
                parent_ind = OPEN.popitem()[0]
            else:
                break
            if parent_ind[0] == goal_ind[0] and parent_ind[1] == goal_ind[1]:
                done = True
                break
            # get neighbors
            children_inds, children_costs = neighbors_func(map, parent_ind)
            # # Get list of children
            # children_inds = self._get_neighbors_inds(parent_ind)
            # # Get list of costs from parent to children
            # children_costs = self._get_parent_children_costs(parent_ind)
            for child_ind, child_cost in zip(children_inds, children_costs):
                if g[child_ind] > g[parent_ind] + child_cost:
                    g[child_ind] = g[parent_ind] + child_cost
                    pred[child_ind[0], child_ind[1], :] = np.array(parent_ind)
                    # This updates if child already in OPEN
                    # and appends to OPEN otherwise
                    OPEN[child_ind] = g[child_ind] + self.eps * np.linalg.norm(np.array(child_ind) - goal)

        # We have found a path
        path = []
        if done:
            pos = goal
            while True:
                path.append(pos)
                if pos[0] == start[0] and pos[1] == start[1]:
                    break
                else:
                    pos = pred[pos[0], pos[1], :]
        path = list(reversed(path))
        path = np.array(path)
        return done, path