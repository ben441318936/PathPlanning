from collections import namedtuple
from unittest import result
import numpy as np

np.set_printoptions(precision=2, suppress=True)
import sys

from time import time

from functools import partial

from MotionModel import DifferentialDriveTorqueInput, DifferentialDriveVelocityInput
from Environment import Environment, Obstacle
from Controller import Controller, PVelocityController, PVelocitySSTorqueController
from Estimator import Estimator, WheelVelocityEstimator, PoseEstimator, FullStateEstimator
from Planner import A_Star_Planner, Planner, SearchBasedPlanner, get_8_neighbors, get_n_grid_neighbors

import pygame
pygame.init()

Offset = namedtuple("Offset", ["top", "bottom", "left", "right"])

class Simulation(object):

    __slots__ = ("_render", "_window_size", "_FPS", "_render_offset", "_center_col_width", 
                 "_environment", "_controller", "_estimator", "_input_noise_var", "_encoder_noise_var", "_planner",
                 "_clock", "_screen", "_color_dict")

    def __init__(self, environment: Environment = None, controller: Controller = None, estimator: Estimator = None,
                 input_noise_var: np.ndarray=None, encoder_noise_var: np.ndarray=None, planner: SearchBasedPlanner = None,
                 render=False, window_size=None, FPS=None, render_offset=(0,0,0,0), center_col_width=0) -> None:
        self._render = render
        self._window_size = window_size
        self._FPS = FPS
        self._render_offset = render_offset # (top,bottom,left,right)
        self._center_col_width = center_col_width
        self._environment : Environment = environment
        self._controller : Controller = controller
        self._estimator : Estimator = estimator
        self._input_noise_var : np.ndarray = input_noise_var
        self._encoder_noise_var : np.ndarray = encoder_noise_var
        self._planner = planner

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

    @property
    def controller(self) -> Controller:
        return self._controller

    @property
    def estimator(self) -> Estimator:
        return self._estimator

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

        # initialize estimator
        self._estimator.init_estimator(self._environment.agent_state)

        while not self._environment.position_out_of_bounds(self._environment.agent_position):
            # get state estimate
            estimated_state = self._estimator.estimate
            estimated_pos = self._environment.motion_model.state_2_position(estimated_state)

            results = self._environment.scan_cone(angle_range=(-np.pi, np.pi), max_range=5, resolution=5/180*np.pi)
            self._planner.update_environment(estimated_pos, results)

            # plan a path
            if not self._planner.path_valid():
                if not self._planner.plan(estimated_pos, self.target):
                    print("Planning failed")
                    print("estimated pos", estimated_pos)
                    print("actual pos", self._environment.agent_position)
                    break
                next_stop = self._planner.take_next_stop()

            if self._planner.path_valid() and np.linalg.norm(estimated_pos - next_stop) < self._planner.map.resolution:
                next_stop =  self._planner.take_next_stop()

            # use state estimate to compute control action to next stop
            control_action = self._controller.control(estimated_state, next_stop)
            # add noise to input
            input_noise = np.random.multivariate_normal(np.zeros((self._input_noise_var.shape[0],)), self._input_noise_var, size=None)
            noisy_input = control_action.copy()
            for (name, i) in zip(self._environment.motion_model.input_names, range(input_noise.shape[0])):
                noisy_input[name] += input_noise[i]
            # apply noisy input to actual robot
            if not self._environment.agent_take_step(input=noisy_input):
                print("Can't take step")
                break
            # get noisy measurments
            # encoder_obs = self._environment.agent_wheel_velocity + np.random.multivariate_normal(np.zeros((self._encoder_noise_var.shape[0],)), self._encoder_noise_var, size=None)
            # run estimator with new measurements
            self._estimator.predict(control_action)
            # self._estimator.update(encoder_obs)

            # debug monitor
            # print("True state:", self._environment.agent_state)
            # print("Esti state:", self._estimator.estimate)
            # print("Erro state:", self._environment.agent_state - self._estimator.estimate)
            # print()

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
        w = 10
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

    input_noise_var = np.diag(np.array([0.1,0.01]))
    encoder_noise_var = 0.005*np.eye(2)

    # Mot = DifferentialDriveTorqueInput(sampling_period=0.01)
    Mot = DifferentialDriveVelocityInput(sampling_period=0.01)

    Env = Environment(motion_model=Mot, target_position=np.array([90,80]))
    # E.agent_heading = np.pi/4
    # Env.add_obstacle(Obstacle(top=20,bottom=10,left=40,right=50))
    # Env.add_obstacle(Obstacle(top=70,bottom=60,left=10,right=70))

    # Env.add_obstacle(Obstacle(top=60,left=53,bottom=40,right=70))

    Env.add_obstacle(Obstacle(top=60,bottom=52,left=52,right=60))
    Env.add_obstacle(Obstacle(top=48,bottom=40,left=52,right=60))

    # Env.agent_position = np.array([50,50])

    # Con = PVelocitySSTorqueController(Mot, KP_V=4, KP_W=100, max_rpm=60, Q=np.diag(np.array([1000,2000])), max_torque=100)
    Con = PVelocityController(Mot)

    # Est = FullStateEstimator(Mot, QN=input_noise_var, RN=encoder_noise_var)
    # Est = WheelVelocityEstimator(Mot, QN=input_noise_var, RN=encoder_noise_var)
    Est = PoseEstimator(Mot, input_noise_var)

    Pla = A_Star_Planner(res=1, neighbor_func=get_8_neighbors, safety_margin=1)

    Sim = Simulation(environment=Env, controller=Con, estimator=Est, planner=Pla,
                     input_noise_var=input_noise_var, encoder_noise_var=encoder_noise_var,
                     render=True, window_size=(1050, 550), FPS=100, render_offset=Offset(50,0,0,0), center_col_width=50)

    Sim.render_frame()

    Sim.check_render_next()

    Sim.run_sim()

    Sim.check_end_sim()