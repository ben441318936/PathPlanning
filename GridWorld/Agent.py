from abc import ABC, abstractmethod

from collections import namedtuple
import numpy as np
from pqdict import pqdict

from enum import IntEnum

from Grid import Grid, Vec2Dir, Dir2Vec, ScanStatus

class MapStatus(IntEnum):
    EMPTY = 0
    OBSTACLE = 1
    TARGET = 2
    AGENT = 3
    BOTH = 4

ExpandWidths = namedtuple("ExpandWidths", ["top", "down", "left", "right"])

VertexInfo = namedtuple("VertexInfo", ["coord", "cost"])

### Surrounding scan functions ###

def scan_8_grid() -> list:
    '''
    Area that the agent can see.
    The coordinates of the cells in this area 
    are returned as list of offsets from the agent.
    '''
    # 9x9 area
    # can be in any order
    area = [np.array([i,j]) for j in range(-1,2) for i in range(-1,2)]
    return area

def scan_circular(radius, ang_res=0.1) -> list:
    '''
    Area that the agent can see, in a circular region.
        radius: radius of the region
        ang_res: angular resolution for generating the points on the circle
    Output is a list of points on the circle, compatible with the raytracing scanning in Grid.
    '''
    angs = np.arange(0, 2*np.pi, ang_res)
    rows = np.around(radius * np.sin(angs)).astype(int)
    cols = np.around(radius * np.cos(angs)).astype(int)
    endpoints = [(rows[i], cols[i]) for i in range(rows.shape[0])]
    endpoints = list(dict.fromkeys(endpoints))
    return endpoints


### Base agent class ###

class Agent(ABC):
    '''
    Base Agent class.
    Define a set of common interfaces that the simulation loop should use.

    The following methods are abstract:
        cone_of_vision
            Defines the area that the agent can see.
        plan_algo
            Defines the planning algorithm used by the agent.

    A few other functions that might be useful to customize:
        update_map
            Updates the map according to scan results. 
            Might need to update other data structures used by the planning algo.
        expand_map
            Expands the map when there are no more possible paths.
            Might need to update other data structures used by the planning algo.
    '''

    def __init__(self, init_map_size=(5,5), max_map_size=None) -> None:
        self._map = np.zeros((init_map_size), dtype=int) # assume everything empty
        self.pos = np.array([init_map_size[0]//2, init_map_size[1]//2], dtype=int) # agent initializes to center of map
        self.init_pos = np.array([init_map_size[0]//2, init_map_size[1]//2], dtype=int)
        self._map[self.pos[0], self.pos[1]] = MapStatus.AGENT
        self.max_map_size = max_map_size
        self.target = None
        self.steps_taken = 0
        self.distance_travelled = 0
        self._path = None
        self._path_ind = 0

    def map_size(self) -> tuple:
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

    @abstractmethod
    def cone_of_vision(self) -> list:
        pass

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

        self.steps_taken += 1
        self.distance_travelled += np.linalg.norm(Dir2Vec[dir])
        return True

    def update_map(self, scan_result) -> None:
        for res in scan_result:
            coord = self.pos + res[0]
            if not self.in_bounds(coord) or coord[0] == 0 or coord[0] == self._map.shape[0]-1 or coord[1] == 0 or coord[1] == self._map.shape[1]-1:
                self.expand_map(self._get_expand_map_widths(coord))
                coord = self.pos + res[0]
            if res[1] == ScanStatus.OBSTACLE:
                self._map[coord[0],coord[1]] = MapStatus.OBSTACLE
            elif res[1] == ScanStatus.TARGET or res[1] == ScanStatus.BOTH:
                self._map[coord[0],coord[1]] = MapStatus.TARGET
            elif res[1] == ScanStatus.EMPTY:
                self._map[coord[0],coord[1]] = MapStatus.EMPTY

    def set_target(self, target_pos) -> None:
        '''
        Set target position relative to the current agent position.
        After setting, self._map[self.target[0], self.target[1]] should be where the target is
        '''
        if self.target is not None:
            self._map[self.target[0], self.target[1]] = MapStatus.EMPTY
        self.target = self.pos + target_pos
        while not self.in_bounds(self.target):
            self.expand_map(self._get_expand_map_widths(self.target))
        if self._map[self.target[0], self.target[1]] == MapStatus.AGENT:
            self._map[self.target[0], self.target[1]] = MapStatus.BOTH
        else:
            self._map[self.target[0], self.target[1]] = MapStatus.BOTH

    def _get_expand_map_widths(self, pos, base_width=(5,5,5,5)) -> ExpandWidths:
        widths = [0,0,0,0]
        if pos[0] < 0:
            widths[0] = -pos[0]
        elif pos[0] >= self._map.shape[0]:
            widths[1] = pos[0] - self._map.shape[0] + 1
        if pos[1] < 0:
            widths[2] = -pos[1]
        elif pos[1] >= self._map.shape[1]:
            widths[3] = pos[1] - self._map.shape[1] + 1
        widths = ExpandWidths(widths[0]+base_width[0], widths[1]+base_width[1], widths[2]+base_width[1], widths[3]+base_width[3])
        return widths

    def expand_map(self, widths: ExpandWidths = ExpandWidths(1,1,1,1) ) -> bool:
        '''
        Used when the current map is too small.
        wdiths (UP, DOWN, LEFT, RIGHT)
        '''
        if self.max_map_size is None or (self._map.shape[0] < self.max_map_size[0] and self._map.shape[1] < self.max_map_size[1]):
            self._map = np.pad(self._map, ((widths.top, widths.down), (widths.left, widths.right)), constant_values=MapStatus.EMPTY)
            pos_offsets = np.array([widths.top, widths.left])
            self.pos = self.pos + pos_offsets
            self.init_pos = self.init_pos + pos_offsets
            self.target = self.target + pos_offsets
            if self._path is not None and self._path.shape[0] != 0:
                self._path += pos_offsets
            return True
        else:
            return False

    def get_neighbors(self, vertex: tuple) -> list:
        infos = []
        for i in range(-1,2):
            for j in range(-1,2):
                pos = np.array(vertex) + np.array([i,j])
                if (not (i == 0 and j == 0)) and pos[0] >= 0 and pos[1] >= 0 and pos[0] < self._map.shape[0] and pos[1] < self._map.shape[1]:
                    if self._map[vertex] != MapStatus.OBSTACLE and self._map[pos[0], pos[1]] != MapStatus.OBSTACLE:
                        cost = np.linalg.norm(np.array([i,j]))
                    else:
                        cost = np.inf
                    coord = tuple(pos)
                    infos.append(VertexInfo(coord, cost))
        return infos

    def get_cost_between_vertices(self, v1: tuple, v2: tuple) -> float:
        if self._map[v1] == MapStatus.OBSTACLE or self._map[v2] == MapStatus.OBSTACLE:
            return np.inf
        else:
            v1 = np.array(v1)
            v2 = np.array(v2)
            dif = np.abs(v1 - v2)
            if dif[0] > 1 or dif[1] > 1:
                return np.inf
            else:
                return np.linalg.norm(dif)

    @abstractmethod
    def plan_algo(self) -> bool:
        pass

    def plan(self) -> bool:
        consecutive_expand = 0

        while True:
            plan_success = self.plan_algo()
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
                if self.expand_map(ExpandWidths(5,5,5,5)):
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


### Weighted A* agent ###

class A_star_agent(Agent):
    '''
    Agent that implements weighted A*.
    '''
    def __init__(self, init_map_size=(5, 5), max_map_size=None, eps=1, vision_func=scan_8_grid) -> None:
        super().__init__(init_map_size, max_map_size)
        # Also define the weight used in A*
        self.eps = eps
        self.num_expanded_nodes = 0
        self.max_queue_size = 0
        self.scan_points = vision_func()

    def cone_of_vision(self) -> list:
        return self.scan_points

    def plan_algo(self) -> bool:
        '''
        Assume that the current map is correct, plan a path to the target.
        Using weighted A* with Euclidean distance as heuristic.
        This heuristic is consistent for all k-connected grid.
        '''
        # Initialize the data structures
        # Labels
        g = np.ones(self._map.shape) * np.inf
        start_ind = tuple(self.pos)
        goal_ind = tuple(self.target)
        g[start_ind] = 0
        # Priority queue for OPEN list
        OPEN = pqdict({})
        OPEN[start_ind] = g[start_ind] + self.eps * np.linalg.norm(self.pos - self.target)
        # Predecessor matrix to keep track of path
        # # create dtype string
        # dtype_string = ",".join(['i' for _ in range(len(map.shape))])
        # pred = np.full(map.shape, -1, dtype=dtype_string)
        pred = np.full((self._map.shape[0], self._map.shape[1], 2), -1, dtype=int)

        done = False

        while not done:
            if len(OPEN) != 0 :
                parent_ind = OPEN.popitem()[0]
            else:
                break
            if parent_ind[0] == goal_ind[0] and parent_ind[1] == goal_ind[1]:
                done = True
                break
            self.num_expanded_nodes += 1
            if len(OPEN) > self.max_queue_size:
                self.max_queue_size = len(OPEN)
            # get neighbors
            infos = self.get_neighbors(parent_ind)
            # # Get list of children
            # children_inds = self._get_neighbors_inds(parent_ind)
            # # Get list of costs from parent to children
            # children_costs = self._get_parent_children_costs(parent_ind)
            for child in infos:
                child_ind = child.coord
                child_cost = child.cost
                if g[child_ind] > g[parent_ind] + child_cost:
                    g[child_ind] = g[parent_ind] + child_cost
                    pred[child_ind[0], child_ind[1], :] = np.array(parent_ind)
                    # This updates if child already in OPEN
                    # and appends to OPEN otherwise
                    OPEN[child_ind] = g[child_ind] + self.eps * np.linalg.norm(np.array(child_ind) - self.target)

        # We have found a path
        path = []
        if done:
            pos = self.target
            while True:
                path.append(pos)
                if pos[0] == self.pos[0] and pos[1] == self.pos[1]:
                    break
                else:
                    pos = pred[pos[0], pos[1], :]
        path = list(reversed(path))
        self._path = np.array(path)
        return done
    

### D* Agent ### 

class D_star_agent(Agent):
    '''
    Agent that implements D* Lite.
    '''
    def __init__(self, init_map_size=(5,5), max_map_size=None, vision_func=scan_8_grid) -> None:
        super().__init__(init_map_size, max_map_size)
        self._init_search_structs()
        self._last_map_change_pos = self.pos
        self.max_queue_size = 0
        self.num_expanded_nodes = 0
        self.scan_points = vision_func()

    def cone_of_vision(self) -> list:
        return self.scan_points

    def set_target(self, target_pos) -> None:
        super().set_target(target_pos)
        self._rhs[tuple(self.target)] = 0
        self._U[tuple(self.target)] = (self._heuristic(self.pos, self.target), 0)

    def _heuristic(self, s1: np.ndarray, s2: np.ndarray) -> float:
        return np.linalg.norm(s1 - s2)

    def _init_search_structs(self) -> None:
        self._U = pqdict({})
        self._k_m = 0
        self._g = np.ones(self._map.shape) * np.inf
        self._rhs = np.ones(self._map.shape) * np.inf
        self._last_map_change_pos = self.pos
        # self._rhs[tuple(self.target)] = 0
        # self._U[tuple(self.target)] = (self._heuristic(self.pos, self.target), 0)

    def _calculate_key(self, s: tuple) -> tuple:
        return (min(self._g[s], self._rhs[s]) + self._heuristic(self.pos, np.array(s)) + self._k_m, min(self._g[s], self._rhs[s]))
    
    def _update_vertex(self, u: tuple) -> None:
        if self._g[u] != self._rhs[u]:
            # updates if already in U
            # inserts if not
            self._U[u] = self._calculate_key(u)
            # print("in update vertex:", u, self._U[u])
        elif self._g[u] == self._rhs[u] and u in self._U:
            del self._U[u]

    def _compute_shortest_path(self) -> None:
        # print("Started search from", tuple(self.pos), "to", tuple(self.target))
        k = 0
        while len(self._U) != 0 and (self._U.topitem()[1] < self._calculate_key(tuple(self.pos)) or self._rhs[tuple(self.pos)] > self._g[tuple(self.pos)]):
            if len(self._U) > self.max_queue_size:
                self.max_queue_size = len(self._U)
            
            self.num_expanded_nodes += 1
            
            k += 1
            u = self._U.topitem()[0]
            key_old = self._U.topitem()[1]
            key_new = self._calculate_key(u)

            # print("At iter", k, u, key_old, key_new, self._g[u], self._rhs[u])

            if key_old < key_new:
                self._U[u] = key_new
            elif self._g[u] > self._rhs[u]:
                self._g[u] = self._rhs[u]
                u = self._U.popitem()[0]
                vertices = self.get_neighbors(u)
                for s in vertices:
                    if s.coord != tuple(self.target):
                        self._rhs[s.coord] = min(self._rhs[s.coord], s.cost + self._g[u])
                    self._update_vertex(s.coord)
            else:
                g_old = self._g[u]
                self._g[u] = np.inf
                vertices = self.get_neighbors(u)
                vertices.append(VertexInfo(u,0))
                for s in vertices:
                    if self._rhs[s.coord] == s.cost + g_old:
                        if s.coord != tuple(self.target):
                            self._rhs[s.coord] = min([v.cost + self._g[v.coord] for v in self.get_neighbors(s.coord)])
                    self._update_vertex(s.coord)

    def plan_algo(self) -> bool:
        self._compute_shortest_path()
        if self._rhs[tuple(self.pos)] == np.inf:
            if len(self._U) == 0:
                # exhausted nodes to search
                # however this could be due to problems with map change
                # reset and search one more time
                self._init_search_structs()
                self._rhs[tuple(self.target)] = 0
                self._U[tuple(self.target)] = (self._heuristic(self.pos, self.target), 0)
                self._compute_shortest_path()
        if self._rhs[tuple(self.pos)] == np.inf:
            # could not find a path
            self._path = np.array([])
            return False
        else:
            incomplete_path = False
            path_dict = {}
            path = []
            pos = self.pos
            while True:
                path.append(pos)
                if tuple(pos) in path_dict:
                    incomplete_path = True
                    break
                else:
                    path_dict[tuple(pos)] = 1
                if pos[0] == self.target[0] and pos[1] == self.target[1]:
                    # found a complete path
                    break
                else:
                    neighbors = self.get_neighbors(tuple(pos))
                    pos = neighbors[0].coord
                    min_cost = neighbors[0].cost + self._g[neighbors[0].coord]
                    for node in neighbors:
                        if node.cost + self._g[node.coord] <= min_cost:
                            min_cost = node.cost + self._g[node.coord]
                            pos = node.coord
                    pos = np.array(pos)

            if incomplete_path:
                # get just one next step
                neighbors = self.get_neighbors(tuple(self.pos))
                pos = neighbors[0].coord
                min_cost = neighbors[0].cost + self._g[neighbors[0].coord]
                for node in neighbors:
                    if node.cost + self._g[node.coord] <= min_cost:
                        min_cost = node.cost + self._g[node.coord]
                        pos = node.coord
                pos = np.array(pos)
                path = [self.pos, pos]

            self._path = np.array(path)
            return True

    def update_map(self, scan_result) -> None:
        # print("Updating map")
        map_changed = False
        for res in scan_result:
            v = self.pos + res[0]
            if not self.in_bounds(v) or v[0] == 0 or v[0] == self._map.shape[0]-1 or v[1] == 0 or v[1] == self._map.shape[1]-1:
                self.expand_map(self._get_expand_map_widths(v))
                v = self.pos + res[0]
            if res[1] == ScanStatus.OBSTACLE:
                if self._map[tuple(v)] != MapStatus.OBSTACLE:
                    # discovered an obstacle we didn't know about
                    if not map_changed:
                        self._k_m += self._heuristic(self.pos, self._last_map_change_pos)
                        self._last_map_change_pos = self.pos
                        map_changed = True
                    # the status of v changed
                    # first change the estimates for edges going from v to its neighbors
                    # we only need to change rhs here, since the costs are computed dynamically
                    # setting the map status will correct future estimates
                    self._rhs[tuple(v)] = np.inf
                    # then change estimates for edges going from u to v
                    # u being the neighbors of v
                    u_s = self.get_neighbors(tuple(v))
                    c_olds = [self.get_cost_between_vertices(u.coord, tuple(v)) for u in u_s]
                    self._map[tuple(v)] = MapStatus.OBSTACLE
                    for u, c_old in zip(u_s, c_olds):
                        if u.coord != tuple(self.target):
                            c_new = self.get_cost_between_vertices(u.coord, tuple(v))
                            if c_old > c_new:
                                # edge cost decreaed, won't happen in current sim assumptions
                                self._rhs[u.coord] = min(self._rhs[u.coord], c_new + self._g[tuple(v)])
                            elif self._rhs[u.coord] == c_old + self._g[tuple(v)]:
                                # we used v to reach u, since v changed, we need to adjust our estimates
                                self._rhs[u.coord] = min([z.cost + self._g[z.coord] for z in self.get_neighbors(u.coord)])
                        self._update_vertex(u.coord)
                else:
                    self._map[tuple(v)] = MapStatus.OBSTACLE
            elif res[1] == ScanStatus.TARGET:
                self._map[tuple(v)] = MapStatus.TARGET
            elif res[1] == ScanStatus.EMPTY:
                self._map[tuple(v)] = MapStatus.EMPTY

    def path_valid(self) -> bool:
        # modified to check if there is still a next action to take
        # because D* does incremental search
        # the next best action might be based on wrong estimates that need to be corrected
        # so it won't lead to the target
        return super().path_valid() and self._path_ind+1 < self._path.shape[0]

    def expand_map(self, widths: ExpandWidths = ExpandWidths(1, 1, 1, 1)) -> bool:
        '''
        Used when the current map is too small.
        wdiths (UP, DOWN, LEFT, RIGHT)
        '''
        if self.max_map_size is None or (self._map.shape[0] < self.max_map_size[0] and self._map.shape[1] < self.max_map_size[1]):
            self._map = np.pad(self._map, ((widths.top, widths.down), (widths.left, widths.right)), constant_values=MapStatus.EMPTY)

            # also pad the arrays storing distance estimates
            self._g = np.pad(self._g, ((widths.top, widths.down), (widths.left, widths.right)), constant_values=np.inf)
            self._rhs = np.pad(self._rhs, ((widths.top, widths.down), (widths.left, widths.right)), constant_values=np.inf)

            pos_offsets = np.array([widths.top, widths.left])
            self.pos = self.pos + pos_offsets
            self.init_pos = self.init_pos + pos_offsets
            self.target = self.target + pos_offsets
            if self._path is not None and self._path.shape[0] != 0:
                self._path += pos_offsets

            # adjust the variable used to keep track of priority bounds
            self._last_map_change_pos += pos_offsets

            # rebuild the priority queue
            temp = [item for item in self._U.items()]
            while len(temp) > 0:
                t = temp.pop()
                # adjust the coordinate
                old_coords = t[0]
                new_coords = (old_coords[0]+pos_offsets[0], old_coords[1]+pos_offsets[1])
                self._U[new_coords] = t[1]

            return True
        else:
            return False
            

if __name__ == "__main__":
    G = Grid((10,10))
    print("Initialization")
    G.print_grid()
    G.place_agent((3,3))
    G.place_target((2,2))
    print("After placement")
    G.print_grid()
    
