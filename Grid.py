import numpy as np
from enum import IntEnum

class GridStatus(IntEnum):
    EMPTY = 0
    WALL = 1
    TARGET = 2
    AGENT = 3
    BOTH = 4
    PREV_AGENT = 5

class ScanStatus(IntEnum):
    OUT_OF_BOUNDS = -2
    OBSTRUCTED = -1
    EMPTY = 0
    WALL = 1
    TARGET = 2
    AGENT = 3
    BOTH = 4

class Direction(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

DirectionDict = {
    (-1,0): Direction.UP,
    (1,0): Direction.DOWN,
    (0,-1): Direction.LEFT,
    (0,1): Direction.RIGHT,
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

    def set_obstacle(self, ind_slices) -> None:
        self._grid[ind_slices] = GridStatus.WALL

    def set_random_obstacle(self, probability) -> None:
        sample = np.random.random_sample(self._grid.shape)
        walls = sample < probability
        self._grid[walls==True] = GridStatus.WALL
        self._grid[walls==False] = GridStatus.EMPTY
        self.place_target(self.target_pos, force=True)
        self.place_agent(self.agent_pos, force=True)

    def agent_reached_target(self) -> None:
        return np.sum(np.abs(self.agent_pos - self.target_pos)) == 0

    def in_bounds(self,coord) -> bool:
        return coord[0] >= 0 and coord[0] < self._grid.shape[0] and coord[1] >= 0 and coord[1] < self._grid.shape[1]

    def not_wall(self,coord) -> bool:
        return self._grid[coord[0],coord[1]] != GridStatus.WALL

    def place_agent(self,pos,force=False) -> bool:
        # force allows us to override walls
        row = pos[0]
        col = pos[1]
        if self.in_bounds((row,col)) and (force or self.not_wall((row,col))):
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
        if self.in_bounds((row,col)) and (force or self.not_wall((row,col))):
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
        if dir == Direction.RIGHT: # right
            coord = self.agent_pos + np.array([0,1])
        elif dir == Direction.DOWN: # down
            coord = self.agent_pos + np.array([1,0])
        elif dir == Direction.LEFT: # left
            coord = self.agent_pos + np.array([0,-1])
        elif dir == Direction.UP: # up
            coord = self.agent_pos + np.array([-1,0])

        if self.in_bounds(coord) and self.not_wall(coord):
            return self.place_agent(coord)
        else:
            return False

    def scan(self,area) -> list:
        '''
        Takes in a list of coordinate offsets, centered around the agent.
        Return the status of each coordinate.
        '''
        result = []
        for offset in area:
            coord = self.agent_pos + offset
            if not self.in_bounds(coord):
                result.append((offset, ScanStatus.OUT_OF_BOUNDS))
            elif self._grid[coord[0],coord[1]] == GridStatus.EMPTY or self._grid[coord[0],coord[1]] == GridStatus.PREV_AGENT:
                result.append((offset, ScanStatus.EMPTY))
            else:
                result.append((offset, self._grid[coord[0],coord[1]]))
            # elif self._grid[coord[0],coord[1]] == GridStatus.AGENT:
            #     result.append((offset, GridStatus.AGENT))
            # elif self._grid[coord[0],coord[1]] == GridStatus.WALL:
            #     result.append((offset, ScanStatus.WALL))
            # elif self._grid[coord[0],coord[1]] == GridStatus.TARGET or self._grid[coord[0],coord[1]] == GridStatus.BOTH:
            #     result.append((offset, ScanStatus.TARGET))
        return result

    def relative_target_pos(self) -> None:
        '''
        Returns the position of the target relative to the agent.
        '''
        return self.target_pos - self.agent_pos

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