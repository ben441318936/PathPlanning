import numpy as np

class Grid(object):
    '''
    Grid that represents the environment.
    Core data structure is a 2D array.
    Each grid cell can be empty, occupied by wall, or has an object in it.
        Empty: 0
        Occupied: 1
        HasTarget: 2
        HasAgent: 3
        HasBoth: 4
    '''

    def __init__(self,grid_size) -> None:
        self._grid = np.zeros((grid_size),dtype=int)
        self.agent_pos = np.array([0,0])
        self.target_pos = np.array([0,0])
        self.place_agent(0,0)
        self.place_target(5,5)

    def place_agent(self,row,col) -> None:
        if self._grid[row,col] != 1:
            if self._grid[row,col] == 2:
                self._grid[row,col] = 4
            else:
                self._grid[row,col] = 3
            self.agent_pos = np.array([row,col])

    def place_target(self,row,col) -> None:
        if self._grid[row,col] != 1:
            if self._grid[row,col] == 3:
                self._grid[row,col] = 4
            else:
                self._grid[row,col] = 2
            self.target_pos = np.array([row,col])

    def agent_move(self,dir) -> bool:
        if dir == 0: # right
            coord = self.agent_pos + np.array([0,1])
        elif dir == 1: # down
            coord = self.agent_pos + np.array([1,0])
        elif dir == 2: # left
            coord = self.agent_pos + np.array([0,-1])
        elif dir == 3: # up
            coord = self.agent_pos + np.array([-1,0])

        if self.not_wall(coord):
            self._grid[self.agent_pos[0],self.agent_pos[1]] = 0
            self.place_agent(coord[0],coord[1])
            return True
        else:
            return False

    def not_wall(self,coord) -> bool:
        return self._grid[coord[0],coord[1]] != 1

    def scan(self,area) -> list:
        '''
        Takes in a list of coordinate offsets, centered around the agent.
        Return the status of each coordinate.
        '''
        result = []
        for offset in area:
            coord = self.agent_pos + offset
            if coord[0] < 0 or coord[0] >= self._grid.shape[0] or coord[1] < 0 or coord[1] >= self._grid.shape[1]:
                result.append((offset, -1))
            else:
                result.append((offset, self._grid[coord[0],coord[1]]))
        return result


    def print_grid(self) -> None:
        print(self._grid)
