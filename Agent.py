import numpy as np

class Agent(object):
    '''
    Agent has a map that updates as it explores.
    Each grid cell can be empty, occupied by wall, or has an object in it.
        Out-of-bounds: -1
        Empty: 0
        Occupied: 1
        HasTarget: 2
    '''
    def __init__(self, map_size) -> None:
        self._map = np.zeros((map_size),dtype=int) # assume everything empty
        self._pos = np.array([map_size[0]//2, map_size[1]//2]) # agent initializes to center of map

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

    def move(self,dir) -> None:
        if dir == 0: # right
            self._pos += np.array([0,1])
        elif dir == 1: # down
            self._pos += np.array([1,0])
        elif dir == 2: # left
            self._pos += np.array([0,-1])
        elif dir == 3: # up
            self._pos += np.array([-1,0])

    def update_map(self, scan_result) -> None:
        for res in scan_result:
            coord = self._pos + res[0]
            if not (coord[0] < 0 or coord[0] >= self._map.shape[0] or coord[1] < 0 or coord[1] >= self._map.shape[1]):
                self._map[coord[0],coord[1]] = res[1]

    def print_map(self) -> None:
        print(self._map)


