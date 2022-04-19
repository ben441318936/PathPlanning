import numpy as np
from pqdict import pqdict

class Weighted_A_star(object):
    def __init__(self) -> None:
        pass

    def _get_neighbors_inds(self, parent_ind) -> list:
        result = []
        parent_pos = np.unravel_index(parent_ind, self._map.shape)
        for i in range(-1,2):
            for j in range(-1,2):
                pos = parent_pos + np.array([i,j])
                if (not (i == 0 and j == 0)) and self.in_bounds(pos):
                    # if self._map[pos[0], pos[1]] != MapStatus.OBSTACLE:
                    result.append(np.ravel_multi_index(pos, self._map.shape))
        return result

    def _get_parent_children_costs(self, parent_ind) -> list:
        result = []
        parent_pos = np.unravel_index(parent_ind, self._map.shape)
        for i in range(-1,2):
            for j in range(-1,2):
                pos = parent_pos + np.array([i,j])
                if (not (i == 0 and j == 0)) and self.in_bounds(pos):
                    if self._map[pos[0], pos[1]] != MapStatus.OBSTACLE:
                        result.append(np.linalg.norm(np.array([i,j])))
                    else:
                        result.append(np.inf)
        return result

    def weighted_A_star(self, map, start, goal, eps=1) -> bool:
        '''
        Assume that the current map is correct, plan a path to the target.
        Using weighted A* with Euclidean distance as heuristic.
        This heuristic is consistent for all k-connected grid.
        '''
        # Initialize the data structures
        # Labels
        g = np.ones((map.shape[0] * map.shape[1])) * np.inf
        start_ind = np.ravel_multi_index(start, map.shape)
        goal_ind = np.ravel_multi_index(goal, map.shape)
        g[start_ind] = 0
        # Priority queue for OPEN list
        OPEN = pqdict({})
        OPEN[start_ind] = g[start_ind] + eps * np.linalg.norm(start - goal)
        # A regular dit for CLOSED list
        # CLOSED = {}
        # Predecessor list to keep track of path
        pred = -np.ones((map.shape[0] * map.shape[1])).astype(int)

        done = False

        while not done:
            if len(OPEN) != 0 :
                parent_ind = OPEN.popitem()[0]
            else:
                break
            # CLOSED[parent_ind] = 0
            if parent_ind == goal_ind:
                done = True
                break
            # Get list of children
            children_inds = self._get_neighbors_inds(parent_ind)
            # Get list of costs from parent to children
            children_costs = self._get_parent_children_costs(parent_ind)
            for j in range(len(children_inds)):
                child_ind = children_inds[j]
                # if (not child_ind == -1) and (not child_ind in CLOSED):
                if (not child_ind == -1):
                    if g[child_ind] > g[parent_ind] + children_costs[j]:
                        g[child_ind] = g[parent_ind] + children_costs[j]
                        pred[child_ind] = parent_ind
                        # This updates if child already in OPEN
                        # and appends to OPEN otherwise
                        child_pos = np.unravel_index(child_ind, self._map.shape)
                        OPEN[child_ind] = g[child_ind] + eps * np.linalg.norm(child_pos - self.target)

        # We have found a path
        path = []
        if done:
            ind = goal_ind
            while True:
                pos = np.unravel_index(ind, self._map.shape)
                path.append(pos)
                if ind == start_ind:
                    break
                else:
                    ind = pred[ind]
        path = list(reversed(path))
        self._path = np.array(path)
        self._path_ind = 0
        return done