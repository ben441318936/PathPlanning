import numpy as np
from pqdict import pqdict

from Grid import Grid, Direction, ScanStatus

class Agent(object):
    '''
    Agent has a map that updates as it explores.
    '''
    def __init__(self, map_size) -> None:
        self._map = np.zeros((map_size), dtype=int) # assume everything empty
        self._pos = np.array([map_size[0]//2, map_size[1]//2], dtype=int) # agent initializes to center of map
        self._target = None
        self._path = None

    def print_map(self) -> None:
        print(self._map)

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

    def move(self, dir, grid:Grid=None) -> bool:
        if grid is None or grid.agent_move(dir):
            if dir == Direction.RIGHT: # right
                self._pos += np.array([0,1])
            elif dir == Direction.DOWN: # down
                self._pos += np.array([1,0])
            elif dir == Direction.LEFT: # left
                self._pos += np.array([0,-1])
            elif dir == Direction.UP: # up
                self._pos += np.array([-1,0])
            return True
        else:
            return False

    def update_map(self, scan_result) -> None:
        for res in scan_result:
            coord = self._pos + res[0]
            if not (coord[0] < 0 or coord[0] >= self._map.shape[0] or coord[1] < 0 or coord[1] >= self._map.shape[1]):
                if res[1] == ScanStatus.WALL or res[1] == ScanStatus.OBSTRUCTED:
                    self._map[coord[0],coord[1]] = ScanStatus.WALL
                elif res[1] == ScanStatus.TARGET or res[1] == ScanStatus.BOTH:
                    self._map[coord[0],coord[1]] = ScanStatus.TARGET
                else:
                    self._map[coord[0],coord[1]] = ScanStatus.EMPTY

    def set_target(self, target_pos) -> None:
        '''
        Set target position relative to the current agent position.
        After setting, self._map[self._target[0], self._target[1]] should be where the target is
        '''
        self._target = self._pos + target_pos
        while self._target[0] < 0 or self._target[0] >= self._map.shape[0] or self._target[1] < 0 or self._target[1] >= self._map.shape[1]:
            self.expand_map()
            self._target = self._pos + target_pos

    def expand_map(self, factor=2) -> None:
        '''
        Used when the current map is too small.
        '''
        new_shape = (int(self._map.shape[0]*factor), int(self._map.shape[1]*factor))
        pad_widths = np.array([(new_shape[0]-self._map.shape[0])//2, (new_shape[1]-self._map.shape[1])//2])
        self._map = np.pad(self._map, ((pad_widths[0], pad_widths[0]), (pad_widths[1], pad_widths[1])), constant_values=ScanStatus.EMPTY)
        self._pos = self._pos + pad_widths

    def _get_neighbors_inds(self, parent_ind) -> list:
        result = []
        parent_pos = np.unravel_index(parent_ind, self._map.shape)
        for i in range(-1,2):
            for j in range(-1,2):
                pos = parent_pos + np.array([i,j])
                if (not (pos[0] == 0 and pos[1] == 0)) and (i == 0 or j == 0) and (pos[0] >= 0 and pos[0] < self._map.shape[0]) and (pos[1] >= 0 and pos[1] < self._map.shape[1]):
                    if self._map[pos[0], pos[1]] != ScanStatus.WALL:
                        result.append(np.ravel_multi_index(pos, self._map.shape))
        return result

    def _get_parent_children_costs(self, parent_ind) -> list:
        result = []
        parent_pos = np.unravel_index(parent_ind, self._map.shape)
        for i in range(-1,2):
            for j in range(-1,2):
                pos = parent_pos + np.array([i,j])
                if (not (pos[0] == 0 and pos[1] == 0)) and (i == 0 or j == 0) and (pos[0] >= 0 and pos[0] < self._map.shape[0]) and (pos[1] >= 0 and pos[1] < self._map.shape[1]):
                    if self._map[pos[0], pos[1]] != ScanStatus.WALL:
                        result.append(1)
        return result
    
    def plan(self) -> bool:
        '''
        Assume that the current map is correct, plan a path to the target.
        Using weighted A* with Euclidean distance as heuristic.
        This heuristic is consistent for all k-connected grid.
        '''
        eps = 1
        # Initialize the data structures
        # Labels
        g = np.ones((self._map.shape[0] * self._map.shape[1])) * np.inf
        start_ind = np.ravel_multi_index(self._pos, self._map.shape)
        target_ind = np.ravel_multi_index(self._target, self._map.shape)
        g[start_ind] = 0
        # Priority queue for OPEN list
        OPEN = pqdict({})
        OPEN[start_ind] = g[start_ind] + eps * np.linalg.norm(self._pos - self._target)
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
                        OPEN[child_ind] = g[child_ind] + eps * np.linalg.norm(child_pos - self._target)

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
        return done

    def get_path_agent_frame(self) -> np.ndarray:
        if self._path is not None:
            return self._path - self._pos
        else:
            return np.array([])

    

if __name__ == "__main__":
    G = Grid((10,10))
    
    G.place_agent(8,8)
    G.place_target(0,4)

    A = Agent((5,5))
    A.set_target(G.relative_target_pos())
    A._map[7,2:10] = ScanStatus.WALL
    print(A._pos)
    print(A._target)
    print(A._map)

    # print("Initial map")
    # A.print_map()
    # A.update_map(G.scan(A.cone_of_vision()))
    # print("After scan")
    # A.print_map()

    # A.move(Direction.UP, grid=G)
    # A.update_map(G.scan(A.cone_of_vision()))
    # print("After move and scan")
    # A.print_map()

    # A.move(Direction.UP, grid=G)
    # A.update_map(G.scan(A.cone_of_vision()))
    # print("After move and scan")
    # A.print_map()

    planning_status = A.plan()
    print("Path found:", planning_status)
    print("Path in agent frame: \n", A.get_path_agent_frame())
    print("Path in world frame: \n", A.get_path_agent_frame() + G.agent_pos)