from tkinter import BOTH
import numpy as np
from pqdict import pqdict

from enum import IntEnum

from Grid import Grid, Direction, DirectionDict, GridStatus, ScanStatus

class MapStatus(IntEnum):
    EMPTY = 0
    OBSTACLE = 1
    TARGET = 2
    AGENT = 3
    BOTH = 4

class Agent(object):
    '''
    Agent has a map that updates as it explores.
    '''
    def __init__(self, init_map_size, max_map_size) -> None:
        self._map = np.zeros((init_map_size), dtype=int) # assume everything empty
        self.pos = np.array([init_map_size[0]//2, init_map_size[1]//2], dtype=int) # agent initializes to center of map
        self._map[self.pos[0], self.pos[1]] = MapStatus.AGENT
        self._max_map_size = max_map_size
        self.target = None
        self._path = None
        self._path_ind = None

    def size(self) -> tuple:
        return self._map.shape

    def get_cell(self,pos) -> MapStatus:
        return self._map[pos[0],pos[1]]

    def print_map(self) -> None:
        for i in range(self._map.shape[0]):
            print(self._map[i])

    def in_bounds(self, coord) -> bool:
        return coord[0] >= 0 and coord[0] < self._map.shape[0] and coord[1] >= 0 and coord[1] < self._map.shape[1]

    def reached_target(self) -> bool:
        return np.sum(np.abs(self.pos - self.target)) == 0

    def cone_of_vision(self) -> list:
        '''
        Area that the agent can see.
        The coordinates of the cells in this area 
        are returned as list of offsets from the agent.
        '''
        # 9x9 area
        # can be in any order
        area = [np.array([i,j]) for j in range(-1,2) for i in range(-1,2) if not (i==0 and j==0)]
        return area

    def move(self, dir) -> bool:
        if self._map[self.pos[0], self.pos[1]] == MapStatus.BOTH:
            self._map[self.pos[0], self.pos[1]] = MapStatus.TARGET
        else:
            self._map[self.pos[0], self.pos[1]] = MapStatus.EMPTY
        if dir == Direction.RIGHT: # right
            self.pos += np.array([0,1])
        elif dir == Direction.DOWN: # down
            self.pos += np.array([1,0])
        elif dir == Direction.LEFT: # left
            self.pos += np.array([0,-1])
        elif dir == Direction.UP: # up
            self.pos += np.array([-1,0])
        if self._map[self.pos[0], self.pos[1]] == MapStatus.OBSTACLE:
            print("Map does not match environment. Invalid move.")
            return False
        elif self._map[self.pos[0], self.pos[1]] == MapStatus.TARGET:
            self._map[self.pos[0], self.pos[1]] = MapStatus.BOTH
        else:
            self._map[self.pos[0], self.pos[1]] = MapStatus.AGENT
        return True

    def update_map(self, scan_result) -> None:
        for res in scan_result:
            coord = self.pos + res[0]
            if self.in_bounds(coord):
                if res[1] == ScanStatus.WALL or res[1] == ScanStatus.OUT_OF_BOUNDS:
                    self._map[coord[0],coord[1]] = MapStatus.OBSTACLE
                elif res[1] == ScanStatus.TARGET:
                    self._map[coord[0],coord[1]] = MapStatus.TARGET
                elif res[1] == ScanStatus.EMPTY:
                    self._map[coord[0],coord[1]] = MapStatus.EMPTY

    def set_target(self, target_pos) -> None:
        '''
        Set target position relative to the current agent position.
        After setting, self._map[self.target[0], self.target[1]] should be where the target is
        '''
        self.target = self.pos + target_pos
        while not self.in_bounds(self.target):
            self.expand_map()
        self._map[self.target[0], self.target[1]] = MapStatus.TARGET

    def expand_map(self, factor=1.5) -> bool:
        '''
        Used when the current map is too small.
        '''
        new_shape = (int(self._map.shape[0]*factor), int(self._map.shape[1]*factor))
        if new_shape[0] < self._max_map_size[0] and new_shape[1] < self._max_map_size[1]:
            pad_widths = np.array([(new_shape[0]-self._map.shape[0])//2, (new_shape[1]-self._map.shape[1])//2])
            self._map = np.pad(self._map, ((pad_widths[0], pad_widths[0]), (pad_widths[1], pad_widths[1])), constant_values=MapStatus.EMPTY)
            self.pos = self.pos + pad_widths
            self.target = self.target + pad_widths
            return True
        else:
            return False

    def _get_neighbors_inds(self, parent_ind) -> list:
        result = []
        parent_pos = np.unravel_index(parent_ind, self._map.shape)
        for i in range(-1,2):
            for j in range(-1,2):
                pos = parent_pos + np.array([i,j])
                if (not (i == 0 and j == 0)) and (i == 0 or j == 0) and self.in_bounds(pos):
                    if self._map[pos[0], pos[1]] != MapStatus.OBSTACLE:
                        result.append(np.ravel_multi_index(pos, self._map.shape))
        return result

    def _get_parent_children_costs(self, parent_ind) -> list:
        result = []
        parent_pos = np.unravel_index(parent_ind, self._map.shape)
        for i in range(-1,2):
            for j in range(-1,2):
                pos = parent_pos + np.array([i,j])
                if (not (i == 0 and j == 0)) and (i == 0 or j == 0) and self.in_bounds(pos):
                    if self._map[pos[0], pos[1]] != MapStatus.OBSTACLE:
                        result.append(1)
        return result

    def weighted_A_star(self) -> bool:
        '''
        Assume that the current map is correct, plan a path to the target.
        Using weighted A* with Euclidean distance as heuristic.
        This heuristic is consistent for all k-connected grid.
        '''
        eps = 1
        # Initialize the data structures
        # Labels
        g = np.ones((self._map.shape[0] * self._map.shape[1])) * np.inf
        start_ind = np.ravel_multi_index(self.pos, self._map.shape)
        target_ind = np.ravel_multi_index(self.target, self._map.shape)
        g[start_ind] = 0
        # Priority queue for OPEN list
        OPEN = pqdict({})
        OPEN[start_ind] = g[start_ind] + eps * np.linalg.norm(self.pos - self.target)
        # A regular list for CLOSED list
        CLOSED = []
        # Predecessor list to keep track of path
        pred = -np.ones((self._map.shape[0] * self._map.shape[1])).astype(int)

        done = False

        while not done:
            if len(OPEN) != 0 :
                parent_ind = OPEN.popitem()[0]
            else:
                break
            CLOSED.append(parent_ind)
            if parent_ind == target_ind:
                done = True
                break
            # Get list of children
            children_inds = self._get_neighbors_inds(parent_ind)
            # Get list of costs from parent to children
            # for a 4-connected grid, cost is 1 for all
            children_costs = self._get_parent_children_costs(parent_ind)
            for j in range(len(children_inds)):
                child_ind = children_inds[j]
                if (not child_ind == -1) and (not child_ind in CLOSED):
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
            ind = target_ind
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

    def plan(self) -> bool:
        while not self.weighted_A_star():
            # if we can't find a path, expand the map and try again
            # this assumes there are other empty spaces outside of current map scope
            if self.expand_map():
                continue
            else:
                return False
        return True

    def get_path(self) -> np.ndarray:
        if self._path is not None:
            return self._path
        else:
            return np.array([])


    def get_path_agent_frame(self) -> np.ndarray:
        if self._path is not None:
            return self._path - self.pos
        else:
            return np.array([])

    def next_action(self) -> np.ndarray:
        '''
        Returns the next action according to the path.
        '''
        next_pos = self._path[self._path_ind+1]
        return DirectionDict[tuple(next_pos-self.pos)]

    def take_next_action(self) -> bool:
        self._path_ind += 1
        next_pos = self._path[self._path_ind]
        return self.move(DirectionDict[tuple(next_pos-self.pos)])

    def path_valid(self) -> bool:
        '''
        Check if next steps in path is valid according to updated map
        '''
        if self._path is None:
            # doesn't have a path yet
            return False
        for i in range(self._path_ind, len(self._path)):
            pos = self._path[i]
            if self._map[pos[0], pos[1]] == MapStatus.OBSTACLE:
                return False
        return True

 