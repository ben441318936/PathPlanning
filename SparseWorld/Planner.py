'''
Implements different planning algorithms.
'''

from abc import ABC, abstractmethod
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
    __slots__ = ("_map", "_neighbor_func", "_margin", "_path_idx_changed")

    def __init__(self, xlim=(0,100), ylim=(0,100), res=1, neighbor_func=get_n_grid_neighbors, safety_margin:int = 0) -> None:
        super().__init__()
        self._map: OccupancyGrid = OccupancyGrid(xlim=xlim, ylim=ylim, res=res)
        self._neighbor_func = neighbor_func
        self._margin: int = np.ceil(safety_margin / self._map.resolution)
        self._path_idx_changed = False

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
            # next stop is inf, something wrong from planning, force a replan
            elif self._path_idx + 1 < self._path.shape[0] and np.sum(self._path[self._path_idx+1]) == np.inf:
                return False
            else:
                self._path_idx_changed = False
                # check the line from current location to next stop
                start = self._map.convert_to_grid_coord(start)
                first_stop_valid = self._collision_free(self._map.get_binary_map(), start, self._path[self._path_idx])
                # if one more stop is available, check it too
                if self._path_idx + 1 < self._path.shape[0]:
                    second_stop_valid = self._collision_free(self._map.get_binary_map(), self._path[self._path_idx], self._path[self._path_idx+1])
                    return first_stop_valid and second_stop_valid
                else:
                    return first_stop_valid
        else:
            return False

    @abstractmethod
    def _plan_algo(binary_map: np.ndarray, start: np.ndarray, target: np.ndarray) -> bool:
        pass

    def _simplify_path(self) -> None:
        if self._path.shape[0] == 1:
            return
        simplified = []
        i = 0
        simplified.append(self._path[i])
        while i < self._path.shape[0]:
            for j in range(i+1, self._path.shape[0]):
                if not self._collision_free(self._map.get_binary_map_safe(margin=self._margin), self._path[i], self._path[j]):
                    simplified.append(self._path[j-1])
                    i = j-1
                    break
            # we got to the end of path with no collision
            if j+1 == self._path.shape[0]:
                simplified.append(self._path[j])
                break
        self._path = np.array(simplified)

    def plan(self, start: np.ndarray, target: np.ndarray) -> bool:
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
            if self._plan_algo(binary_map, start, target):
                self._simplify_path()
                self._path_idx_changed = True
                self._path_idx = 0
                return True
        else:
            print("Planning failed with discrete cells")
            print("Start:", start)
            print("Start map status:", self._map.get_status(start))
            print("Target:", target)
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

    def __init__(self, xlim=(0,100), ylim=(0,100), res:float=1, neighbor_func=get_n_grid_neighbors, safety_margin:int = 0, eps:int=1, heuristic=np.linalg.norm) -> None:
        super().__init__(xlim=xlim, ylim=ylim, res=res, neighbor_func=neighbor_func, safety_margin=safety_margin)
        self._eps: int = eps
        self._heuristic = heuristic

    def _plan_algo(self, binary_map: np.ndarray, start: np.ndarray, target: np.ndarray) -> bool:
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