from collections import namedtuple
import numpy as np
np.set_printoptions(precision=2)
import sys

from time import time

from functools import partial

from MotionModels import DifferentialDrive, DifferentialDriveVelocityInput
from Environment import Environment, Obstacle
from Controller import Controller, PVelocityControl, PVelocitySSTorqueControl

import pygame
pygame.init()

Offset = namedtuple("Offset", ["top", "bottom", "left", "right"])

class Simulation(object):

    __slots__ = ("_render", "_window_size", "_FPS", "_render_offset", "_center_col_width", "_env", "_goal", "_controller", 
                    "_clock", "_screen", "_color_dict")

    def __init__(self, env: Environment, controller: Controller, goal: np.ndarray=None, render=False, window_size=None, FPS=None, render_offset=(0,0,0,0), center_col_width=0) -> None:
        self._render = render
        self._window_size = window_size
        self._FPS = FPS
        self._render_offset = render_offset # (top,bottom,left,right)
        self._center_col_width = center_col_width
        self._env : Environment = env
        self._goal : np.ndarray = goal
        self._controller : Controller = controller

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
    def env(self) -> Environment:
        return self._env

    @property
    def goal(self) -> np.ndarray:
        return self._goal

    @property
    def controller(self) -> Controller:
        return self._controller

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

        while not self._env.position_out_of_bounds(self._env.agent_position):
            # input_torque = simple_control(self.env.motion_model, self.env.agent_state, self.goal)
            input_torque = self._controller.control(self._env.agent_state, self._goal)
            # input_torque = np.array([-10,10])
            if not self._env.agent_take_step(input=input_torque):
                print("Can't take step")
                break
            # print(self.env.agent_state)
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
            return round(env_rect.bottom - (coord / self.env.env_size[1] * env_rect.height))
        def scale_x(coord: float) -> int:
            return round(env_rect.left + (coord / self.env.env_size[0] * env_rect.width))
        
        def scale_obs(obs: Obstacle) -> Obstacle:
            return Obstacle(
                top = scale_y(obs.top),
                bottom = scale_y(obs.bottom),
                left = scale_x(obs.left),
                right = scale_x(obs.right)
            )

        # draw obstacles
        for obs in self.env.obstacles:
            scaled_obs = scale_obs(obs)
            obs_rect = pygame.Rect(scaled_obs.left, scaled_obs.top, scaled_obs.right - scaled_obs.left, scaled_obs.bottom - scaled_obs.top)
            pygame.draw.rect(self._screen, self._color_dict["brown"], obs_rect)

        # draw agent
        agent_heading = -self.env.agent_heading # negative here because pygames has a different coordinate system
        w = 10
        points = np.array([[0,  -w,  w,  -w],
                            [0, -w,  0,  w]])
        rotated = np.array([[np.cos(agent_heading), -np.sin(agent_heading)],
                            [np.sin(agent_heading), np.cos(agent_heading)]]) @ points
        rotated = np.around(rotated).astype(int)
        agent_pos = np.array([scale_x(self.env.agent_position[0]), scale_y(self.env.agent_position[1])])
        final_pts = agent_pos.reshape((2,1)) + rotated
        pygame.draw.polygon(self._screen, self._color_dict["red"], [final_pts[:,0], final_pts[:,1], final_pts[:,2], final_pts[:,3]])

        # draw goal
        scaled_goal = (scale_x(self.goal[0]), scale_y(self.goal[1]))
        pygame.draw.circle(self._screen, self._color_dict["green"], scaled_goal, 3)



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
    M = DifferentialDrive(sampling_period=0.01)
    E = Environment(motion_model=M)
    # E.agent_heading = np.pi/4
    # E.add_obstacle(Obstacle(top=20,bottom=10,left=40,right=50))
    # E.add_obstacle(Obstacle(top=70,bottom=60,left=10,right=70))

    # E.agent_heading = 2*np.pi

    C = PVelocitySSTorqueControl(M, Q=np.diag(np.array([1000,2000])))

    S = Simulation(E, C, goal=np.array([10,40]), render=True, window_size=(1050, 550), FPS=100, render_offset=Offset(50,0,0,0), center_col_width=50)

    S.render_frame()

    S.check_render_next()

    S.run_sim()

    S.check_end_sim()