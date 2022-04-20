from collections import namedtuple
from cv2 import Algorithm
import numpy as np
from pqdict import pqdict

from enum import IntEnum

from Grid import Grid, Direction, Vec2Dir, Dir2Vec, GridStatus, ScanStatus

from Algorithms import Weighted_A_star

class MapStatus(IntEnum):
    EMPTY = 0
    OBSTACLE = 1
    TARGET = 2
    AGENT = 3
    BOTH = 4

ExpandWidths = namedtuple("ExpandWidths", ["top", "down", "left", "right"])

class Agent(object):
    '''
    Agent has a map that updates as it explores.
    '''
    def __init__(self, planner=None, init_map_size=(5,5)) -> None:
        self._map = np.zeros((init_map_size), dtype=int) # assume everything empty
        self.pos = np.array([init_map_size[0]//2, init_map_size[1]//2], dtype=int) # agent initializes to center of map
        self.init_pos = np.array([init_map_size[0]//2, init_map_size[1]//2], dtype=int)
        self._map[self.pos[0], self.pos[1]] = MapStatus.AGENT
        # self._max_map_size = max_map_size
        self.target = None
        self._path = None
        self._path_ind = None
        if planner is None:
            self.planner = Weighted_A_star(eps=10)
        else:
            self.planner = planner

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
        area = [np.array([i,j]) for j in range(-1,2) for i in range(-1,2)]
        return area

    def move(self, dir) -> bool:
        if self._map[self.pos[0], self.pos[1]] == MapStatus.BOTH:
            self._map[self.pos[0], self.pos[1]] = MapStatus.TARGET
        else:
            self._map[self.pos[0], self.pos[1]] = MapStatus.EMPTY

        self.pos += Dir2Vec[dir]
        if self._map[self.pos[0], self.pos[1]] == MapStatus.OBSTACLE:
            print("Invalid move.")
            self.pos -= Dir2Vec[dir]
            return False
        elif self._map[self.pos[0], self.pos[1]] == MapStatus.TARGET:
            self._map[self.pos[0], self.pos[1]] = MapStatus.BOTH
        else:
            self._map[self.pos[0], self.pos[1]] = MapStatus.AGENT
        return True

    def update_map(self, scan_result) -> None:
        for res in scan_result:
            coord = self.pos + res[0]
            while not self.in_bounds(coord):
                self.expand_map(self._get_expand_map_widths(coord))
                coord = self.pos + res[0]
            if res[1] == ScanStatus.OBSTACLE:
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
            self.expand_map(self._get_expand_map_widths(self.target))
        self._map[self.target[0], self.target[1]] = MapStatus.TARGET

    def _get_expand_map_widths(self, pos) -> ExpandWidths:
        widths = [0,0,0,0]
        if pos[0] < 0:
            widths[0] = -pos[0]
        elif pos[0] >= self._map.shape[0]:
            widths[1] = pos[0] - self._map.shape[0] + 1
        if pos[1] < 0:
            widths[2] = -pos[1]
        elif pos[1] >= self._map.shape[1]:
            widths[3] = pos[1] - self._map.shape[1] + 1
        widths = ExpandWidths(widths[0], widths[1], widths[2], widths[3])
        return widths

    def expand_map(self, widths: ExpandWidths = ExpandWidths(1,1,1,1) ) -> bool:
        '''
        Used when the current map is too small.
        wdiths (UP, DOWN, LEFT, RIGHT)
        '''
        self._map = np.pad(self._map, ((widths.top, widths.down), (widths.left, widths.right)), constant_values=MapStatus.EMPTY)
        pos_offsets = np.array([widths.top, widths.left])
        self.pos = self.pos + pos_offsets
        self.init_pos = self.init_pos + pos_offsets
        self.target = self.target + pos_offsets
        if self._path is not None and self._path.shape[0] != 0:
            self._path += pos_offsets
        return True

    def plan(self) -> bool:
        consecutive_expand = 0

        while True:
            plan_success, self._path = self.planner.plan(self._map, self.pos, self.target, get_8_neighbors)
            self._path_ind = 0
            if not plan_success:
                if np.sum(self._map[0,:]) + np.sum(self._map[-1,:]) + np.sum(self._map[:,0]) + np.sum(self._map[:,-1]) == 0:
                    print("Planning failed, expanding won't help.")
                    return False
                    # if we can't find a path, expand the map and try again
                    # this assumes there are other empty spaces outside of current map scope
                if consecutive_expand > 0:
                    print("Planning still failed after expanding.")
                    return False
                if self.expand_map():
                    consecutive_expand += 1
                else:
                    return False
            else:
                return True

    def get_path(self) -> np.ndarray:
        if self._path is not None:
            return self._path
        else:
            return np.array([])

    def get_path_agent_frame(self) -> np.ndarray:
        if self._path is not None and self._path.shape[0] != 0:
            return self._path - self.pos
        else:
            return np.array([])

    def next_action(self) -> np.ndarray:
        '''
        Returns the next action according to the path.
        '''
        next_pos = self._path[self._path_ind+1]
        return Vec2Dir[tuple(next_pos-self.pos)]

    def take_next_action(self) -> bool:
        self._path_ind += 1
        next_pos = self._path[self._path_ind]
        return self.move(Vec2Dir[tuple(next_pos-self.pos)])

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

### Auxillary

def get_8_neighbors(map, parent) -> tuple:
    inds = []
    costs = []
    for i in range(-1,2):
        for j in range(-1,2):
            pos = np.array(parent) + np.array([i,j])
            if (not (i == 0 and j == 0)) and pos[0] >= 0 and pos[1] >= 0 and pos[0] < map.shape[0] and pos[1] < map.shape[1]:
                if map[pos[0], pos[1]] != MapStatus.OBSTACLE:
                    costs.append(np.linalg.norm(np.array([i,j])))
                else:
                    costs.append(np.inf)
                inds.append(tuple(pos))
    return inds, costs