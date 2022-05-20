'''
Implements different planning algorithms.
'''

from abc import ABC, abstractmethod
from time import time
from tracemalloc import stop
import numpy as np
from typing import List

from Environment import ScanResult
from Map import Map, GridStatus, GridMap, OccupancyGrid, raytrace

from pqdict import pqdict

from collections import namedtuple

from functools import partial

class Planner(ABC):
    '''
    Basic planner interface.
    '''

    __slots__ = ("_path", "_path_idx")

    def __init__(self) -> None:
        self._path: np.ndarray = None
        self._path_idx: int = None

    @property
    def path(self) -> np.ndarray:
        if self._path is None:
            return np.array([])
        else:
            return self._path

    def path_valid(self) -> bool:
        '''
        Check if current planned path is still valid according to what we know.
        '''
        if self._path is None:
            # doesn't have a path yet
            return False
        return True

    @abstractmethod
    def plan(self, start: np.ndarray, target: np.ndarray) -> bool:
        '''
        Implements high level planning logic.
        '''
        pass

    def next_stop(self) -> np.ndarray:
        '''
        Return the next stop in the planned path.
        '''
        self._path_idx += 1
        return self._path[self._path_idx]

    @abstractmethod
    def update_environment(self, scan_start: np.ndarray, scan_results: List[ScanResult]) -> None:
        '''
        Updates the planner's version of the environment.
        Could be updating a map or a list of obstacles/landmarks.
        '''
        pass

VertexInfo = namedtuple("VertexInfo", ["coord", "cost"])

def get_8_neighbors(map: np.ndarray, vertex: tuple) -> List[VertexInfo]:
    infos = []
    for i in range(-1,2):
        for j in range(-1,2):
            pos = np.array(vertex) + np.array([i,j])
            if (not (i == 0 and j == 0)) and pos[0] >= 0 and pos[1] >= 0 and pos[0] < map.shape[0] and pos[1] < map.shape[1]:
                if map[vertex] != GridStatus.OBSTACLE and map[pos[0], pos[1]] != GridStatus.OBSTACLE:
                    cost = np.linalg.norm(np.array([i,j]))
                else:
                    cost = np.inf
                coord = tuple(pos)
                infos.append(VertexInfo(coord, cost))
    return infos

def get_n_grid_neighbors(map: np.ndarray, vertex: tuple, n: int = 3) -> List[VertexInfo]:
    '''
    Get neighbors in a n x n grid. Using a binary map.

    n should be an odd interger greater than or equal to 3.

    Ex: n=3, get the 8 neighbors in a 3x3 grid.
    '''
    if n < 3:
        n = 3
    if n % 2 == 0:
        n += 1
    infos = []
    for i in range(-(n-1)//2,(n-1)//2+1):
        for j in range(-(n-1)//2,(n-1)//2+1):
            pos = np.array(vertex) + np.array([i,j])
            if (not (i == 0 and j == 0)) and pos[0] >= 0 and pos[1] >= 0 and pos[0] < map.shape[0] and pos[1] < map.shape[1]:
                ray_xx, ray_yy = raytrace(vertex[0], vertex[1], pos[0], pos[1])
                if np.sum(map[ray_xx, ray_yy] == GridStatus.OBSTACLE) == 0:
                    cost = np.linalg.norm(np.array([i,j]))
                else:
                    cost = np.inf
                coord = tuple(pos)
                infos.append(VertexInfo(coord, cost))
    return infos
    

class SearchBasedPlanner(Planner):
    '''
    Planner based on discretizing the environment and searching the resulting graph.
    '''
    __slots__ = ("_map", "_neighbor_func", "_margin", "_path_idx_changed", "_total_planning_time", "_total_update_time")

    def __init__(self, map, neighbor_func=get_8_neighbors, safety_margin:int = 0) -> None:
        super().__init__()
        self._map: OccupancyGrid = map
        self._neighbor_func = neighbor_func
        self._margin: int = np.ceil(safety_margin / self._map.resolution)
        self._path_idx_changed = False
        self._total_planning_time = 0
        self._total_update_time = 0

    def __del__(self):
        print("Total planning time", self._total_planning_time)
        print("Total update time", self._total_update_time)

    @property
    def map(self) -> OccupancyGrid:
        return self._map

    def update_environment(self, scan_start: np.ndarray, scan_results: List[ScanResult]) -> None:
        return self._map.update_map(scan_start, scan_results)

    def next_stop(self) -> np.ndarray:
        self._stop_idx_changed = True
        return self._map.convert_to_world_coord(super().next_stop())

    def _collision_free(self, map: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> bool:
        ray_xx, ray_yy = raytrace(v1[0], v1[1], v2[0], v2[1])
        if np.sum(map[ray_xx, ray_yy] == GridStatus.OBSTACLE) == 0:
            return True
        else:
            return False

    def path_valid(self, start: np.ndarray) -> bool:
        if super().path_valid():
            # map didn't change, and the next stop didn't change
            # no need to check for collision
            if self._map.old_map_valid and not self._path_idx_changed:
                return True
            else:
                self._path_idx_changed = False
                # check the line from current location to next stop
                start = self._map.convert_to_grid_coord(start)
                first_stop_valid = self._collision_free(self._map.get_binary_map(), start, self._path[self._path_idx])
                # return first_stop_valid
                # checking the next next stop gives a better path
                # but more frequent planning
                if self._path_idx + 1 < self._path.shape[0]:
                    second_stop_valid = self._collision_free(self._map.get_binary_map(), self._path[self._path_idx], self._path[self._path_idx+1])
                    return first_stop_valid and second_stop_valid
                else:
                    return first_stop_valid
        else:
            return False

    @abstractmethod
    def _plan_algo(self, binary_map: np.ndarray, start: np.ndarray, target: np.ndarray) -> bool:
        # print("In _plan_algo", start, target)
        pass

    def _simplify_path(self, binary_map: np.ndarray) -> None:
        # print("In _simplify_path")
        simplified = []
        i = 0
        simplified.append(self._path[i])
        while i < self._path.shape[0]:
            for j in range(i+1, self._path.shape[0]):
                if not self._collision_free(binary_map, self._path[i], self._path[j]):
                    if j-2 >= i+1:
                        simplified.append(self._path[j-2])
                        i = j-2
                    else:
                        simplified.append(self._path[j-1])
                        i = j-1
                    break
            # we got to the end of path with no collision
            if j+1 == self._path.shape[0]:
                simplified.append(self._path[j])
                break
            # somehow there is no collision free connections to the next stop
            # likely because noise put us too close to the obstacle
            # then just follow the next stop without simplifying
            else:
                simplified.append(self._path[i+1])
                i += 1
        self._path = np.array(simplified)

    def plan(self, start: np.ndarray, target: np.ndarray) -> bool:
        # print("In planning", start, target)
        start = self._map.convert_to_grid_coord(start)
        target = self._map.convert_to_grid_coord(target)
        if start[0] == target[0] and start[1] == target[1]:
            self._path = np.array([start])
            return True
        if self._margin == 0:
            binary_map = self._map.get_binary_map()
        else:
            binary_map = self._map.get_binary_map_safe(margin = self._margin)
        # sometimes the safety margin grows into the starting location
        # but it is actually safe
        if self._map.get_status(start) == GridStatus.EMPTY:
            binary_map[start[0], start[1]] = GridStatus.EMPTY
            t = time()
            if self._plan_algo(binary_map, start, target):
                self._total_planning_time += time() - t
                # print("Raw path:", self._path)
                # self._simplify_path(binary_map)
                self._path_idx_changed = True
                self._path_idx = 0
                return True
            else:
                print("Failed to find a path")
                print("Start:", start)
                print("Start:", start)
                return False
        else:
            print("Trying to plan starting from an occupied cell")
            print("Start:", start)
            print("Start map status:", self._map.get_status(start))
            print("Start:", start)
            return False
    
    def next_stop(self) -> np.ndarray:
        # wstart and target are in the same postion
        # or we have reached the end, keep outputting the target position
        if self._path_idx + 1 >= self._path.shape[0]:
            self._path_idx -= 1
        # output the next stop, force a collision check next time we check path valid
        else:
            self._path_idx_changed = True
        return super().next_stop()


class A_Star_Planner(SearchBasedPlanner):
    '''
    Planner than implements weighted A*.
    '''

    __slots__ = ("_eps", "_heuristic")

    def __init__(self, map, neighbor_func=get_8_neighbors, safety_margin:int = 0, eps:int=1, heuristic=np.linalg.norm) -> None:
        super().__init__(map, neighbor_func=neighbor_func, safety_margin=safety_margin)
        self._eps: int = eps
        self._heuristic = heuristic

    def _plan_algo(self, binary_map: np.ndarray, start: np.ndarray, target: np.ndarray) -> bool:
        super()._plan_algo(binary_map, start, target)
        # Initialize the data structures
        # Labels
        g = np.ones(binary_map.shape) * np.inf
        start_ind = tuple(start)
        goal_ind = tuple(target)
        g[start_ind] = 0
        # Priority queue for OPEN list
        OPEN = pqdict({})
        OPEN[start_ind] = g[start_ind] + self._eps * self._heuristic(start - target)
        # Predecessor matrix to keep track of path
        # # create dtype string
        # dtype_string = ",".join(['i' for _ in range(len(map.shape))])
        # pred = np.full(map.shape, -1, dtype=dtype_string)
        pred = np.full((binary_map.shape[0], binary_map.shape[1], 2), -1, dtype=int)

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
            infos = self._neighbor_func(binary_map, parent_ind)
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
                    OPEN[child_ind] = g[child_ind] + self._eps * self._heuristic(np.array(child_ind) - target)

        # We have found a path
        path = []
        if done:
            pos = target
            while True:
                path.append(pos)
                if pos[0] == start[0] and pos[1] == start[1]:
                    break
                else:
                    pos = pred[pos[0], pos[1], :]
            path = list(reversed(path))
        self._path = np.array(path)
        return done

    def update_environment(self, scan_start: np.ndarray, scan_results: List[ScanResult]) -> None:
        t = time()
        res = super().update_environment(scan_start, scan_results)
        self._total_update_time += time() - t
        return res


class D_Star_Planner(SearchBasedPlanner):
    '''
    Planner that implements D* Lite.
    More efficient when there is frequent replanning.
    '''

    __slots__ = ("_U", "_k_m", "_g", "_rhs", "_last_map_change_pos", "_pos", "_target", "_heuristic")

    def __init__(self, map, neighbor_func=get_8_neighbors, safety_margin: int = 0, heuristic=np.linalg.norm) -> None:
        super().__init__(map, neighbor_func, safety_margin)
        self._heuristic = heuristic
        self._init_search_structs()

    def _init_search_structs(self):
        self._U = pqdict({})
        self._k_m = 0
        self._g = np.ones(self._map.shape) * np.inf
        self._rhs = np.ones(self._map.shape) * np.inf
        self._pos = None # most recent planning start position
        self._target = None
        self._last_map_change_pos = None # position at which the last map change occurred

    def _set_start_target(self, start: np.ndarray, target: np.ndarray) -> None:
        self._pos = start
        if self._target is None:
            self._target = target
            self._rhs[tuple(self._target)] = 0
            self._U[tuple(self._target)] = (self._heuristic(start - self._target), 0)
        elif np.sum(self._target != target) != 0:
            # target changed since we last performed a search
            # reset all structures
            self._init_search_structs()
            self._target = target
            self._rhs[tuple(self._target)] = 0
            self._U[tuple(self._target)] = (self._heuristic(start - self._target), 0)
        if self._last_map_change_pos is None:
            self._last_map_change_pos = start
        
    def _plan_algo(self, binary_map: np.ndarray, start: np.ndarray, target: np.ndarray) -> bool:
        '''
        Possible to generate incomplete paths with D*, i.e. paths that doesn't reach the target.
        Such paths are [start, next, next, ..., inf]
        '''
        super()._plan_algo(binary_map, start, target)
        self._set_start_target(start, target)
        self._compute_shortest_path(binary_map)
        if self._rhs[tuple(self._pos)] == np.inf:
            if len(self._U) == 0:
                # exhausted nodes to search
                # however this could be due to problems with map change
                # reset and search one more time
                self._init_search_structs()
                self._set_start_target(start, target)
                self._compute_shortest_path(binary_map)
        if self._rhs[tuple(self._pos)] == np.inf:
            # could not find a path
            self._path = np.array([])
            return False
        else:
            incomplete_path = False
            path_dict = {}
            path = []
            pos = self._pos
            while True:
                path.append(pos)
                if tuple(pos) in path_dict:
                    incomplete_path = True
                    break
                else:
                    path_dict[tuple(pos)] = 1
                if pos[0] == self._target[0] and pos[1] == self._target[1]:
                    # found a complete path
                    break
                else:
                    neighbors = self._neighbor_func(binary_map, tuple(pos))
                    coords = [z.coord for z in neighbors]
                    total_cost = np.array([z.cost + self._g[z.coord] for z in neighbors])
                    pos = np.array(coords[np.argmin(total_cost)])

            if incomplete_path:
                # get just one next step
                neighbors = self._neighbor_func(binary_map, tuple(self._pos))
                pos = neighbors[0].coord
                min_cost = neighbors[0].cost + self._g[neighbors[0].coord]
                for node in neighbors:
                    if node.cost + self._g[node.coord] <= min_cost:
                        min_cost = node.cost + self._g[node.coord]
                        pos = node.coord
                pos = np.array(pos)
                path = [self._pos, pos]

            self._path = np.array(path)
            return True

    def _calculate_key(self, s: tuple) -> tuple:
        return (min(self._g[s], self._rhs[s]) + self._heuristic(self._pos - np.array(s)) + self._k_m, min(self._g[s], self._rhs[s]))

    def _update_vertex(self, u: tuple) -> None:
        if self._g[u] != self._rhs[u]:
            # updates if already in U
            # inserts if not
            self._U[u] = self._calculate_key(u)
            # print("in update vertex:", u, self._U[u])
        elif self._g[u] == self._rhs[u] and u in self._U:
            del self._U[u]
    
    def _compute_shortest_path(self, binary_map: np.ndarray) -> None:
        # print("Started search from", tuple(self.pos), "to", tuple(self.target))
        # k = 0
        while len(self._U) != 0 and (self._U.topitem()[1] < self._calculate_key(tuple(self._pos)) or self._rhs[tuple(self._pos)] > self._g[tuple(self._pos)]):
            # if len(self._U) > self.max_queue_size:
            #     self.max_queue_size = len(self._U)
                      
            # k += 1
            u = self._U.topitem()[0]
            key_old = self._U.topitem()[1]
            key_new = self._calculate_key(u)

            # print("At iter", k, u, key_old, key_new, self._g[u], self._rhs[u])

            if key_old < key_new:
                self._U[u] = key_new
            elif self._g[u] > self._rhs[u]:
                self._g[u] = self._rhs[u]
                u = self._U.popitem()[0]
                vertices = self._neighbor_func(binary_map, u)
                for s in vertices:
                    if s.coord != tuple(self._target):
                        self._rhs[s.coord] = min(self._rhs[s.coord], s.cost + self._g[u])
                    self._update_vertex(s.coord)
            else:
                g_old = self._g[u]
                self._g[u] = np.inf
                vertices = self._neighbor_func(binary_map, u)
                vertices.append(VertexInfo(u,0))
                for s in vertices:
                    if self._rhs[s.coord] == s.cost + g_old:
                        if s.coord != tuple(self._target):
                            self._rhs[s.coord] = min([v.cost + self._g[v.coord] for v in self._neighbor_func(binary_map, s.coord)])
                    self._update_vertex(s.coord)

    def update_environment(self, scan_start: np.ndarray, scan_results: List[ScanResult]) -> None:
        t = time()
        # get a copy of the old map in order to check how the map changed later
        # but we only do this if we know where the target is
        # i.e. we have planned at least once
        # otherwise all cost estimated are inf, no need to update
        if self._target is not None:
            if self._margin == 0:
                old_map = self._map.get_binary_map()
            else:
                old_map = self._map.get_binary_map_safe(margin = self._margin)

        # use parent function, propagates to the map object
        super().update_environment(scan_start, scan_results)

        # map object signals that the occupancy map has changed
        if self._target is not None and not self._map.old_map_valid:
            self._pos = self._map.convert_to_grid_coord(scan_start[0:2])
            self._k_m += self._heuristic(self._pos - self._last_map_change_pos)
            self._last_map_change_pos = self._pos

            # get the changed cells
            if self._margin == 0:
                new_map = self._map.get_binary_map()
            else:
                new_map = self._map.get_binary_map_safe(margin = self._margin)
            dif = new_map - old_map # -1, 0, 1
            changed_idxs = np.nonzero(dif)

            # for each changed vertex
            for i in range(changed_idxs[0].shape[0]):
                # the status of v changed
                v = (changed_idxs[0][i], changed_idxs[1][i])
                increased = dif[v] == 1
                # first change the estimates for edges going from v to its neighbors
                # we only need to change rhs here, since the costs are computed dynamically
                # the updated map will correct future estimates
                if increased:
                    # v changed from empty cell to obstacle
                    # v now has inf cost
                    self._rhs[v] = np.inf
                else:
                    # v changed from obstacle -> empty cell
                    # cost of v is min over all neighbors: (cost to go to neighbor z + cost from z to target)
                    self._rhs[v] = min([z.cost + self._g[z.coord] for z in self._neighbor_func(new_map, v)])
                # then change estimates for edges going from u to v
                # u being the neighbors of v
                u_s_old = self._neighbor_func(old_map, v)
                u_s_new = self._neighbor_func(new_map, v)
                for u_old, u_new in zip(u_s_old, u_s_new):
                    u = u_old.coord
                    if u != tuple(self._target):
                        c_old = u_old.cost
                        c_new = u_new.cost
                        if c_old > c_new:
                            # edge cost decreased, this means we corrected some obstacle positions
                            self._rhs[u] = min(self._rhs[u], c_new + self._g[v])
                        elif self._rhs[u] == c_old + self._g[v]:
                            # we used v to reach u, since v changed, we adjust our estimates for u
                            self._rhs[u] = min([z.cost + self._g[z.coord] for z in self._neighbor_func(new_map, u)])
                        # edge cost from u to v increased, but we didn't use v to get to u, so no change
                    self._update_vertex(u)
        self._total_update_time += time() - t

    def _simplify_path(self, binary_map: np.ndarray) -> None:
        super()._simplify_path(binary_map)
        if np.sum(np.abs(self._path[-1] - self._target)) > 0:
            self._path = np.vstack((self._path, self._target))



    
        




    
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from MotionModel import DifferentialDriveTorqueInput
    from Environment import Environment, Obstacle

    M = DifferentialDriveTorqueInput(sampling_period=0.1)
    E = Environment(motion_model=M, target_position=np.array([90,80]))

    E.agent_heading = 0

    # E.add_obstacle(Obstacle(top=60,bottom=52,left=52,right=60))
    # E.add_obstacle(Obstacle(top=48,bottom=40,left=52,right=60))

    E.add_obstacle(Obstacle(top=60,left=53,bottom=40,right=70))

    # MAP = OccupancyGrid(xlim=(0,100), ylim=(0,100), res=1)
    P = A_Star_Planner(xlim=(-10,100), ylim=(-10,100), res=1, neighbor_func=partial(get_n_grid_neighbors, n=3))

    results = E.scan_cone(angle_range=(-np.pi/2, np.pi/2), max_range=5, resolution=1/180*np.pi)
    P.update_environment(E.agent_position, results)

    P.plan(E.agent_position, E.target_position)

    print(P._path)

    plt.figure()
    plt.plot(P.path[:,0], P.path[:,1])
    plt.show()