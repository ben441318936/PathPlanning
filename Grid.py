import numpy as np
from enum import IntEnum

from skimage.draw import line as raytrace


class GridStatus(IntEnum):
    EMPTY = 0
    OBSTACLE = 1
    TARGET = 2
    AGENT = 3
    BOTH = 4
    PREV_AGENT = 5

class ScanStatus(IntEnum):
    OUT_OF_BOUNDS = -2
    OBSTRUCTED = -1
    EMPTY = 0
    OBSTACLE = 1
    TARGET = 2
    AGENT = 3
    BOTH = 4

class Direction(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    UP_LEFT = 4
    UP_RIGHT = 5
    DOWN_LEFT = 6
    DOWN_RIGHT = 7

Vec2Dir = {
    (-1,0): Direction.UP,
    (1,0): Direction.DOWN,
    (0,-1): Direction.LEFT,
    (0,1): Direction.RIGHT,
    (-1,-1): Direction.UP_LEFT,
    (-1,1): Direction.UP_RIGHT,
    (1,-1): Direction.DOWN_LEFT,
    (1,1): Direction.DOWN_RIGHT,
}

Dir2Vec = {
    Direction.UP: np.array([-1,0]),
    Direction.DOWN: np.array([1,0]),
    Direction.LEFT: np.array([0,-1]),
    Direction.RIGHT: np.array([0,1]),
    Direction.UP_LEFT: np.array([-1,-1]),
    Direction.UP_RIGHT: np.array([-1,1]),
    Direction.DOWN_LEFT: np.array([1,-1]),
    Direction.DOWN_RIGHT: np.array([1,1]),
}


class Grid(object):
    '''
    Grid that represents the environment.
    Core data structure is a 2D array.
    '''
    def __init__(self,grid_size) -> None:
        self._grid = np.zeros((grid_size),dtype=int)
        self.agent_pos = None
        self.target_pos = None

    def size(self) -> tuple:
        return self._grid.shape

    def get_cell(self,pos) -> GridStatus:
        return self._grid[pos]

    def agent_reached_target(self) -> None:
        return np.sum(np.abs(self.agent_pos - self.target_pos)) == 0

    def in_bounds(self,coord) -> bool:
        return coord[0] >= 0 and coord[0] < self._grid.shape[0] and coord[1] >= 0 and coord[1] < self._grid.shape[1]

    def not_obstacle(self,coord) -> bool:
        return self._grid[coord[0],coord[1]] != GridStatus.OBSTACLE

    def set_obstacle(self, ind_slices) -> None:
        self._grid[ind_slices] = GridStatus.OBSTACLE

    def fill_random_grid(self, probability) -> None:
        sample = np.random.random_sample(self._grid.shape)
        obs = sample < probability
        self._grid[obs==True] = GridStatus.OBSTACLE
        self._grid[obs==False] = GridStatus.EMPTY
        # self.place_target(self.target_pos, force=True)
        # self.place_agent(self.agent_pos, force=True)

    def set_random_target(self) -> None:
        self.target_pos = np.zeros((2), dtype=int)
        self.target_pos[0] = np.random.randint(0,self._grid.shape[0])
        self.target_pos[1] = np.random.randint(0,self._grid.shape[1])
        self.place_target(self.target_pos, force=True)

    def set_random_agent(self) -> None:
        self.agent_pos = np.zeros((2), dtype=int)
        self.agent_pos[0] = np.random.randint(0,self._grid.shape[0])
        self.agent_pos[1] = np.random.randint(0,self._grid.shape[1])
        self.place_agent(self.agent_pos, force=True)

    def place_agent(self,pos,force=False) -> bool:
        # force allows us to override obstacles
        row = pos[0]
        col = pos[1]
        if self.in_bounds((row,col)) and (force or self.not_obstacle((row,col))):
            if self.agent_pos is not None:
                self._grid[self.agent_pos[0], self.agent_pos[1]] = GridStatus.PREV_AGENT
            if self._grid[row,col] == GridStatus.TARGET:
                self._grid[row,col] = GridStatus.BOTH
            else:
                self._grid[row,col] = GridStatus.AGENT
            self.agent_pos = np.array([row,col])
            return True
        else:
            return False

    def place_target(self,pos,force=False) -> None:
        row = pos[0]
        col = pos[1]
        if self.in_bounds((row,col)) and (force or self.not_obstacle((row,col))):
            if self.target_pos is not None:
                self._grid[self.target_pos[0], self.target_pos[1]] = GridStatus.EMPTY
            if self._grid[row,col] == GridStatus.AGENT:
                self._grid[row,col] = GridStatus.BOTH
            else:
                self._grid[row,col] = GridStatus.TARGET
            self.target_pos = np.array([row,col])
            return True
        else:
            return False

    def agent_move(self,dir) -> bool:
        coord = self.agent_pos + np.array(Dir2Vec[dir])
        if self.in_bounds(coord) and self.not_obstacle(coord):
            return self.place_agent(coord)
        else:
            return False

    def scan_cells(self,area) -> list:
        '''
        Takes in a list of coordinate offsets as np 2-vector, centered around the agent.
        Return the status of each coordinate.
        The status can be EMPTY or OBSTACLE or TARGET or BOTH
        '''
        result = []
        for offset in area:
            coord = self.agent_pos + offset
            if not self.in_bounds(coord):
                result.append((offset, ScanStatus.OBSTACLE))
            elif self._grid[coord[0],coord[1]] == GridStatus.EMPTY or self._grid[coord[0],coord[1]] == GridStatus.PREV_AGENT:
                result.append((offset, ScanStatus.EMPTY))
            elif self._grid[coord[0],coord[1]] == GridStatus.AGENT:
                result.append((offset, ScanStatus.AGENT))
            elif self._grid[coord[0],coord[1]] == GridStatus.TARGET:
                result.append((offset, ScanStatus.TARGET))
            elif self._grid[coord[0],coord[1]] == GridStatus.BOTH:
                result.append((offset, ScanStatus.BOTH))
        return result

    def scan_cone(self,cone_ends) -> list:
        '''
        Takes in a list of coordinates, representing the endpoints of a cone centered around the agent.
        Return the status of each coordinate.
        The status can be EMPTY or OBSTACLE or AGENT or TARGET or BOTH.
        
        This works by performing ray tracing, starting from the agent pos, ending at each endpoint.
        The ray will stop at the first cell is that is TARGET or OBSTACLE.
        All cells before this cell will be EMPTY.
        Return nothing for cells after the obstacle since agent can not see it.
        '''
        result = {}
        for end_offset in cone_ends:
            endpoints = self.agent_pos + end_offset
            ray_cc, ray_rr = raytrace(self.agent_pos[0], self.agent_pos[1], endpoints[0], endpoints[1])
            for c, r in zip(ray_cc, ray_rr):
                coord = (c,r)
                offset = (c - self.agent_pos[0], r - self.agent_pos[1])
                if not self.in_bounds(coord) or self._grid[coord[0],coord[1]] == GridStatus.OBSTACLE:
                    result[coord] = (offset, ScanStatus.OBSTACLE)
                    break
                elif self._grid[coord[0],coord[1]] == GridStatus.TARGET:
                    result[coord] = (offset, ScanStatus.TARGET)
                    break
                elif self._grid[coord[0],coord[1]] == GridStatus.BOTH:
                    result[coord] = (offset, ScanStatus.BOTH)
                    break
                elif self._grid[coord[0],coord[1]] == GridStatus.EMPTY or self._grid[coord[0],coord[1]] == GridStatus.PREV_AGENT:
                    result[coord] = (offset, ScanStatus.EMPTY)
                elif self._grid[coord[0],coord[1]] == GridStatus.AGENT:
                    result[coord] = (offset, ScanStatus.AGENT)
                
        result_list = [val for val in result.values()]
        return result_list

    def relative_target_pos(self) -> None:
        '''
        Returns the position of the target relative to the agent.
        '''
        return self.target_pos - self.agent_pos

    def translate_path_to_world_frame(self, path) -> np.ndarray:
        return path + self.agent_pos

    def print_grid(self) -> None:
        for i in range(self._grid.shape[0]):
            print(self._grid[i])


if __name__ == "__main__":
    G = Grid((10,10))
    print("Initialization")
    G.print_grid()
    G.place_agent((3,3))
    G.place_target((2,2))
    print("After placement")
    G.print_grid()
    G.agent_move(Direction.UP)
    print("After agent move")
    G.print_grid()

    print(G.scan_cone([np.array([0,-2])]))