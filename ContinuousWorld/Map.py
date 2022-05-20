'''
Implements different map storages that an agent/planner/estimator will use.

Includes a Map interface for all map objects.
'''

import numpy as np
from abc import ABC, abstractmethod
from enum import IntEnum

from skimage.draw import line
from skimage.morphology import square, dilation

from Environment import ScanResult
from typing import List

def raytrace(start_x: int, start_y: int, end_x: int, end_y: int) -> tuple:
    ray_yy, ray_xx = line(start_y, start_x, end_y, end_x)
    return ray_xx, ray_yy

class GridStatus(IntEnum):
    EMPTY = 0
    OBSTACLE = 1
    TARGET = 2
    OOB = 3 # out of bounds

class Map(ABC):
    '''
    Basic interface for map objects.
    '''

    __slots__ = ("_map", "_xlim", "_ylim")

    def __init__(self, xlim=(0,10), ylim=(0,10)) -> None:
        self._xlim = xlim
        self._ylim = ylim
        self._map = None

    @abstractmethod
    def get_status(self, coord: np.ndarray) -> GridStatus:
        '''
        Checks the status of some floating point coordinate.
        '''
        pass

    @abstractmethod
    def update_map(self, scan_start: np.ndarray, scans: List[ScanResult]) -> None:
        pass


class GridMap(Map):
    '''
    Map based on grid-like discretization of the environment.
    '''
    __slots__ = ()

    def __init__(self, xlim=(0, 10), ylim=(0, 10)) -> None:
        super().__init__(xlim, ylim)

    @abstractmethod
    def convert_to_grid_coord(self, coord: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def convert_to_world_coord(self, coord: np.ndarray) -> np.ndarray:
        pass

class OccupancyGrid(GridMap):
    '''
    An occupancy map based on evenly spaced discretization.
    '''

    __slots__ = ("_res", "_n_cells", "_LOG_ODDS", "_LOG_ODDS_LIM", "_old_map_valid", "_binary_map")

    def __init__(self, xlim=(-5, 5), ylim=(-5, 5), res=1, log_odds=np.log(4), log_odds_lim_factor=10) -> None:
        super().__init__(xlim, ylim)
        self._LOG_ODDS = log_odds
        self._LOG_ODDS_LIM = log_odds * log_odds_lim_factor
        self._res = res
        n_cells_x = int(np.ceil((xlim[1]-xlim[0]) / res + 1))
        n_cells_y = int(np.ceil((ylim[1]-ylim[0]) / res + 1))
        self._n_cells = (n_cells_x, n_cells_y)
        self._map = np.zeros(self._n_cells) # need data type double to handle log odds
        self._binary_map = np.zeros(self._n_cells, dtype=int)
        self._old_map_valid = False

    @property
    def old_map_valid(self) -> bool:
        return self._old_map_valid

    @property
    def shape(self) -> tuple:
        return self._n_cells

    @property
    def resolution(self) -> float:
        return self._res

    def convert_to_grid_coord(self, coord: np.ndarray) -> np.ndarray:
        '''
        Converts floating point coordinates into integer grid coordinates used to index the cells.
        '''
        map_center = ((self._xlim[1] - self._xlim[0]) / 2, (self._ylim[1] - self._ylim[0]) / 2)
        center_cell = ((self._n_cells[0]-1)/2, (self._n_cells[1]-1)/2) # these are always ints
        out_coord = np.zeros((2), dtype=int)
        for i in range(coord.shape[0]):
            offset = coord[i] - map_center[i]
            if np.abs(offset) <= self._res / 2:
                out_coord[i] = center_cell[i]
            else:
                out_coord[i] = center_cell[i] + np.sign(offset) * np.ceil((np.abs(offset) - self._res/2) / self._res)
        return out_coord

    def convert_to_world_coord(self, coord: np.ndarray) -> np.ndarray:
        '''
        Convert grid cell indices into world coordinates.
        '''
        map_center = ((self._xlim[1] - self._xlim[0]) / 2, (self._ylim[1] - self._ylim[0]) / 2)
        center_cell = ((self._n_cells[0]-1)/2, (self._n_cells[1]-1)/2) # these are always ints
        offset = coord - center_cell
        return self._res * offset + map_center

    def get_status(self, coord: np.ndarray) -> GridStatus:
        return self._binary_map[coord[0], coord[1]]

    def update_map(self, scan_pose: np.ndarray, scans: List[ScanResult], max_range: int = 5) -> None:
        '''
        scan_pose should be floating point world coordinates
        corresponding to the pose of the robot when the LIDAR was done
        '''
        ray_start = self.convert_to_grid_coord(scan_pose[0:2])
        center_heading = scan_pose[2]

        # scans is N x (ang,rng)
        angs = np.array([scan.angle for scan in scans]) + center_heading
        rngs = np.array([scan.range for scan in scans])

        # process those that got inf range, i.e. did not hit an obstacle at all
        angs_inf = angs[rngs == np.inf].reshape((-1,1)) # (N,1)

        endpoints = ray_start + max_range * np.hstack((np.cos(angs_inf), np.sin(angs_inf))) # (N,2)
        for i in range(endpoints.shape[0]):
            endpoint = endpoints[i,:]
            endpoint = self.convert_to_grid_coord(endpoint)
            ray_xx, ray_yy = raytrace(ray_start[0], ray_start[1], endpoint[0], endpoint[1])
            self._map[ray_xx, ray_yy] -= self._LOG_ODDS

        # process those that got finite range, i.e. hit an obstacle
        angs_hit = angs[rngs<np.inf].reshape((-1,1)) # (N,1)
        rngs_hit = rngs[rngs<np.inf].reshape((-1,1)) # (N,1)

        endpoints = scan_pose[0:2] + rngs_hit * np.hstack((np.cos(angs_hit), np.sin(angs_hit))) # (N,2)
        for i in range(endpoints.shape[0]):
            endpoint = endpoints[i,:]
            endpoint = self.convert_to_grid_coord(endpoint)
            ray_xx, ray_yy = raytrace(ray_start[0], ray_start[1], endpoint[0], endpoint[1])
            last_point = (ray_xx[-1], ray_yy[-1])
            self._map[last_point[0], last_point[1]] += self._LOG_ODDS
            ray_xx = ray_xx[0:-1]
            ray_yy = ray_yy[0:-1]
            self._map[ray_xx, ray_yy] -= self._LOG_ODDS

        np.clip(self._map, -self._LOG_ODDS_LIM, self._LOG_ODDS_LIM, out=self._map)

        new_binary_map = self._compute_binary_map()
        self._old_map_valid = np.sum(self._binary_map != new_binary_map) == 0
        if not self._old_map_valid:
            self._binary_map = new_binary_map

    def _compute_binary_map(self) -> np.ndarray:
        return 1 * (self._map > 0)

    def get_binary_map(self) -> np.ndarray:
        return self._binary_map

    def get_binary_map_safe(self, margin: int = 1) -> np.ndarray:
        binary_map = self.get_binary_map()
        binary_map_safe = dilation(binary_map, square(int(margin*2+1)))
        return binary_map_safe



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from MotionModel import DifferentialDriveTorqueInput
    from Environment import Environment, Obstacle

    M = DifferentialDriveTorqueInput(sampling_period=0.1)
    E = Environment(motion_model=M, target_position=np.array([90,50]))

    E.agent_heading = 0

    E.add_obstacle(Obstacle(top=60,bottom=52,left=52,right=60))
    E.add_obstacle(Obstacle(top=48,bottom=40,left=52,right=60))

    results = E.scan_cone(angle_range=(-np.pi/2, np.pi/2), max_range=5, resolution=5/180*np.pi)

    MAP = OccupancyGrid(xlim=(0,100), ylim=(0,100), res=1)
    MAP.update_map(E.agent_position, results)

    # map = MAP.get_binary_map_safe(margin=2)
    map = MAP.get_binary_map()

    map = np.rot90(map)

    plt.imshow(map, cmap="gray")
    plt.show()

    # plt.figure()
    # for res in results:
    #     ang = res.angle
    #     rng = res.range
    #     start = E.agent_position
    #     end = start + rng * np.array([np.cos(ang), np.sin(ang)])
    #     plt.plot([start[0], end[0]], [start[1], end[1]])
    #     plt.axis("equal")
    # plt.show()


    
        

    



