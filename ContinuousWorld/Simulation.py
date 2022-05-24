from collections import namedtuple
import numpy as np

from Map import OccupancyGrid

np.set_printoptions(precision=2, suppress=True)
import sys

from time import time

from functools import partial

from MotionModel import DifferentialDriveTorqueInput, DifferentialDriveVelocityInput
from Environment import Environment, Obstacle
from Controller import Controller, PLinearSSTorqueController, PVelocityController, PVelocitySSTorqueController
from Estimator import Estimator, WheelVelocityEstimator, PoseEstimator, FullStateEstimator
from Planner import A_Star_Planner, D_Star_Planner, Planner, SearchBasedPlanner, get_8_neighbors, get_n_grid_neighbors
from Agent import OccupancyGrid, OccupancyGridAgent

import pygame
pygame.init()

Offset = namedtuple("Offset", ["top", "bottom", "left", "right"])

class Simulation(object):

    __slots__ = ("_render", "_window_size", "_FPS", "_render_offset", "_center_col_width", 
                 "_environment", "_agent", "_input_noise_var", "_encoder_noise_var",
                 "_clock", "_screen", "_color_dict")

    def __init__(self, environment: Environment = None, agent: OccupancyGridAgent = None,
                 input_noise_var: np.ndarray=None, encoder_noise_var: np.ndarray=None,
                 render=False, window_size=None, FPS=None, render_offset=(0,0,0,0), center_col_width=0) -> None:
        self._render = render
        self._window_size = window_size
        self._FPS = FPS
        self._render_offset = render_offset # (top,bottom,left,right)
        self._center_col_width = center_col_width
        self._environment : Environment = environment
        self._agent : OccupancyGridAgent = agent
        self._input_noise_var : np.ndarray = input_noise_var
        self._encoder_noise_var : np.ndarray = encoder_noise_var

        if render and (window_size is None or FPS is None):
            print("Render set to True, but no render parameters given.")
            sys.exit()
        if render:
            self._clock = pygame.time.Clock()
            self._screen = pygame.display.set_mode(window_size)
            self._color_dict = {"black": (0,0,0), 
                                "red": (255,0,0), 
                                "white": (255,255,255), 
                                "gray": (128,128,128),
                                "blue": (0,0,255),
                                "purple": (128,0,128),
                                "green": (0,255,0),
                                "yellow": (255,255,0),
                                "brown": (205,133,63),
                                "turquoise": (100,200,255),
                            }

    @property
    def environment(self) -> Environment:
        return self._environment

    @property
    def target(self) -> np.ndarray:
        return self._environment.target_position

    def check_render_next(self) -> None:
        started = False
        while not started:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: 
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN: 
                    started = True

    def check_end_sim(self) -> None:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: 
                    sys.exit()

    def run_sim(self, manual=False) -> None:
        # render
        if self._render:
            self.render_frame()
            if manual:
                self.check_render_next()

        # init agent state using info from environment
        self._agent.set_agent_state(self._environment.agent_state)
        self._agent.set_target_position(self._environment.target_position)

        while not self._environment.position_out_of_bounds(self._environment.agent_position) and not self._agent.reached_target():
            # get observations
            obs = {}
            obs["LIDAR"] = self._environment.scan_cone(self._agent.scan_angles, self._agent.scan_max_range)
            obs["ENCODER"] = self._environment.agent_wheel_velocity + np.random.multivariate_normal(np.zeros((self._encoder_noise_var.shape[0],)), self._encoder_noise_var, size=None)
            self._agent.process_observation(obs)

            # get control action from agent
            control_action = self._agent.control()

            # add noise to input
            input_noise = np.random.multivariate_normal(np.zeros((self._input_noise_var.shape[0],)), self._input_noise_var, size=None)
            noisy_input = control_action.copy()
            for (name, i) in zip(self._environment.motion_model.input_names, range(input_noise.shape[0])):
                noisy_input[name] += input_noise[i]
            # apply noisy input to actual robot
            if not self._environment.agent_take_step(input=noisy_input):
                print("Control action invalid for environment")
                break

            # render
            if self._render:
                self.render_frame()
                if manual:
                    self.check_render_next()

    def draw_env(self, env_rect: pygame.Rect) -> None:
        '''
        Draw the obstacles and agents, scaled to fit in env_rect
        (x,y) from env will be converted to (left+x,bot-y) to fit pygame conventions
        '''

        pygame.draw.rect(self._screen, self._color_dict["gray"], env_rect, 1)

        # the scaled coordinates follows pygame conventions
        # (right > left), (bottom > top)
        def scale_y(coord: float) -> int:
            return round(env_rect.bottom - (coord / self._environment.env_size[1] * env_rect.height))
        def scale_x(coord: float) -> int:
            return round(env_rect.left + (coord / self._environment.env_size[0] * env_rect.width))
        
        def scale_obs(obs: Obstacle) -> Obstacle:
            return Obstacle(
                top = scale_y(obs.top),
                bottom = scale_y(obs.bottom),
                left = scale_x(obs.left),
                right = scale_x(obs.right)
            )

        # draw obstacles
        for obs in self._environment.obstacles:
            scaled_obs = scale_obs(obs)
            obs_rect = pygame.Rect(scaled_obs.left, scaled_obs.top, scaled_obs.right - scaled_obs.left, scaled_obs.bottom - scaled_obs.top)
            pygame.draw.rect(self._screen, self._color_dict["brown"], obs_rect)

        # draw agent
        agent_heading = -self._environment.agent_heading # negative here because pygames has a different coordinate system
        w = 5
        points = np.array([[0,  -w,  w,  -w],
                            [0, -w,  0,  w]])
        rotated = np.array([[np.cos(agent_heading), -np.sin(agent_heading)],
                            [np.sin(agent_heading), np.cos(agent_heading)]]) @ points
        rotated = np.around(rotated).astype(int)
        agent_pos = np.array([scale_x(self._environment.agent_position[0]), scale_y(self._environment.agent_position[1])])
        final_pts = agent_pos.reshape((2,1)) + rotated
        pygame.draw.polygon(self._screen, self._color_dict["red"], [final_pts[:,0], final_pts[:,1], final_pts[:,2], final_pts[:,3]])

        # draw target
        scaled_target = (scale_x(self.target[0]), scale_y(self.target[1]))
        pygame.draw.circle(self._screen, self._color_dict["green"], scaled_target, 3)

    def render_frame(self) -> None:
        if not self._render:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()

        # clear window
        self._screen.fill(self._color_dict["black"])

        sub_figure_width = (self._window_size[0] - (self._render_offset.left + self._render_offset.right) - self._center_col_width) // 2
        sub_fugure_height = self._window_size[1] - (self._render_offset.top + self._render_offset.bottom)

        # draw the environment
        env_rect = pygame.Rect(self._render_offset.left, self._render_offset.top, 
                                sub_figure_width, 
                                sub_fugure_height)
        self.draw_env(env_rect)

        # grid status label text
        env_label_rect = pygame.Rect(0, 0, (self._window_size[0]-self._center_col_width)//2, self._render_offset.top)
        self._screen.fill(self._color_dict["black"], env_label_rect)
        my_font = pygame.font.SysFont("Times New Roman", 30)
        my_text = my_font.render("Environment Status", True, self._color_dict["white"])
        my_rect = my_text.get_rect()
        width = my_rect.width
        height = my_rect.height
        self._screen.blit(my_text, (env_label_rect.centerx - width//2, env_label_rect.centery - height//2))

        # draw center column
        pygame.draw.rect(self._screen, self._color_dict["black"], 
                pygame.Rect((self._window_size[0]-self._center_col_width)//2, 0, self._center_col_width, self._window_size[1]))

        # # draw the agent's map
        # map_rect = pygame.Rect(sub_figure_width + self.center_col_width,
        #                         self.render_offset.top, 
        #                         sub_figure_width, 
        #                         sub_fugure_height)
        # self.draw_map(map_rect)

        # # map status label text
        # map_label_rect = pygame.Rect((self.window_size[0]-self.center_col_width)//2 + self.center_col_width, 0,
        #                                 (self.window_size[0]-self.center_col_width)//2, self.render_offset.top)
        # self.screen.fill(self.color_dict["black"], map_label_rect)
        # my_font = pygame.font.SysFont("Times New Roman", 30)
        # my_text = my_font.render("Agent Map", True, self.color_dict["white"])
        # my_rect = my_text.get_rect()
        # width = my_rect.width
        # height = my_rect.height
        # self.screen.blit(my_text, (map_label_rect.centerx - width//2, map_label_rect.centery - height//2))

        pygame.display.flip()
        self._clock.tick(self._FPS)


if __name__ == "__main__":

    input_noise_var = np.diag(np.array([0.1,0.01]))*0
    encoder_noise_var = 0.005*np.eye(2)

    # torque input
    Mot = DifferentialDriveTorqueInput(sampling_period=0.01)
    Con = PVelocitySSTorqueController(Mot, KP_V=4, KP_W=20, max_rpm=60, Q=np.diag(np.array([1000,2000])), max_torque=100)
    # Con = PLinearSSTorqueController(Mot, KP_V=4, max_rpm=60, Q=np.diag(np.array([100,100,0.01])), max_torque=100)
    Est = FullStateEstimator(Mot, QN=input_noise_var, RN=encoder_noise_var)
    # Est = WheelVelocityEstimator(Mot, QN=input_noise_var, RN=encoder_noise_var)

    # velocity input
    # Mot = DifferentialDriveVelocityInput(sampling_period=0.01)
    # Con = PVelocityController(Mot)
    # Est = PoseEstimator(Mot, input_noise_var)

    Map = OccupancyGrid(xlim=(0,100), ylim=(0,100), res=1)
    
    # Pla = A_Star_Planner(Map, neighbor_func=get_8_neighbors, safety_margin=1)

    # D* is more efficient if there is a lot of replanning
    Pla = D_Star_Planner(Map, neighbor_func=get_8_neighbors, safety_margin=1)

    Age = OccupancyGridAgent(Mot, Pla, Con, Est, Map)

    Env = Environment(motion_model=Mot, target_position=np.array([10,50]))
    # E.agent_heading = np.pi/4
    # Env.add_obstacle(Obstacle(top=20,bottom=10,left=40,right=50))
    # Env.add_obstacle(Obstacle(top=70,bottom=60,left=10,right=70))

    # Env.add_obstacle(Obstacle(top=60,left=53,bottom=40,right=70))

    # Env.add_obstacle(Obstacle(top=60,bottom=53,left=5,right=40))
    # Env.add_obstacle(Obstacle(top=60,bottom=52,left=60,right=64))
    # Env.add_obstacle(Obstacle(top=48,bottom=40,left=5,right=40))
    # Env.add_obstacle(Obstacle(top=70,left=50,bottom=40,right=70))

    Env.agent_position = np.array([50,50])

    Sim = Simulation(environment=Env, agent=Age,
                     input_noise_var=input_noise_var, encoder_noise_var=encoder_noise_var,
                     render=True, window_size=(1050, 550), FPS=100, render_offset=Offset(50,0,0,0), center_col_width=50)

    Sim.render_frame()

    Sim.check_render_next()

    Sim.run_sim()

    Sim.check_end_sim()