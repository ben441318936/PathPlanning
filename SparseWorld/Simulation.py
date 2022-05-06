from collections import namedtuple
import numpy as np
np.set_printoptions(precision=2)
import sys

from time import time

from functools import partial

from MotionModels import DifferentialDrive, DifferentialDriveVelocityInput
from Environment import Environment, Obstacle
from Controller import Controller, DoublePDControl, PVelocityControl

import pygame
pygame.init()

Offset = namedtuple("Offset", ["top", "bottom", "left", "right"])

class Simulation(object):
    def __init__(self, env: Environment, controller: Controller, goal: np.ndarray=None, render=False, window_size=None, FPS=None, render_offset=(0,0,0,0), center_col_width=0) -> None:
        self.render = render
        self.window_size = window_size
        self.FPS = FPS
        self.render_offset = render_offset # (top,bottom,left,right)
        self.center_col_width = center_col_width
        self.env : Environment = env
        self.goal : np.ndarray = goal
        self.controller :Controller = controller

        if render and (window_size is None or FPS is None):
            print("Render set to True, but no render parameters given.")
            sys.exit()
        if render:
            self.clock = pygame.time.Clock()
            self.screen = pygame.display.set_mode(window_size)
            self.color_dict = {"black": (0,0,0), 
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
        if self.render:
            self.render_frame()
            if manual:
                self.check_render_next()  

        while not self.env.position_out_of_bounds(self.env.agent_position):
            # input_torque = simple_control(self.env.motion_model, self.env.agent_state, self.goal)
            input_torque = self.controller.control(self.env.agent_state, self.goal)
            # input_torque = np.array([-10,10])
            if not self.env.agent_take_step(input=input_torque):
                print("Can't take step")
                break

            # render
            if self.render:
                self.render_frame()
                if manual:
                    self.check_render_next()

    def draw_env(self, env_rect: pygame.Rect) -> None:
        '''
        Draw the obstacles and agents, scaled to fit in env_rect
        (x,y) from env will be converted to (left+x,bot-y) to fit pygame conventions
        '''

        pygame.draw.rect(self.screen, self.color_dict["gray"], env_rect, 1)

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
        for obs in self.env.Obstacles:
            scaled_obs = scale_obs(obs)
            obs_rect = pygame.Rect(scaled_obs.left, scaled_obs.top, scaled_obs.right - scaled_obs.left, scaled_obs.bottom - scaled_obs.top)
            pygame.draw.rect(self.screen, self.color_dict["brown"], obs_rect)

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
        pygame.draw.polygon(self.screen, self.color_dict["red"], [final_pts[:,0], final_pts[:,1], final_pts[:,2], final_pts[:,3]])

        # draw goal
        scaled_goal = (scale_x(self.goal[0]), scale_y(self.goal[1]))
        pygame.draw.circle(self.screen, self.color_dict["green"], scaled_goal, 3)



    def render_frame(self) -> None:
        if not self.render:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()

        # clear window
        self.screen.fill(self.color_dict["black"])

        sub_figure_width = (self.window_size[0] - (self.render_offset.left + self.render_offset.right) - self.center_col_width) // 2
        sub_fugure_height = self.window_size[1] - (self.render_offset.top + self.render_offset.bottom)

        # draw the environment
        env_rect = pygame.Rect(self.render_offset.left, self.render_offset.top, 
                                sub_figure_width, 
                                sub_fugure_height)
        self.draw_env(env_rect)

        # grid status label text
        env_label_rect = pygame.Rect(0, 0, (self.window_size[0]-self.center_col_width)//2, self.render_offset.top)
        self.screen.fill(self.color_dict["black"], env_label_rect)
        my_font = pygame.font.SysFont("Times New Roman", 30)
        my_text = my_font.render("Environment Status", True, self.color_dict["white"])
        my_rect = my_text.get_rect()
        width = my_rect.width
        height = my_rect.height
        self.screen.blit(my_text, (env_label_rect.centerx - width//2, env_label_rect.centery - height//2))

        # draw center column
        pygame.draw.rect(self.screen, self.color_dict["black"], 
                pygame.Rect((self.window_size[0]-self.center_col_width)//2, 0, self.center_col_width, self.window_size[1]))

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
        self.clock.tick(self.FPS)


if __name__ == "__main__":
    M = DifferentialDriveVelocityInput(sampling_period=0.01)
    E = Environment(motion_model=M)
    # E.agent_heading = np.pi/4
    # E.add_obstacle(Obstacle(top=20,bottom=10,left=40,right=50))
    # E.add_obstacle(Obstacle(top=70,bottom=60,left=10,right=70))

    # E.agent_heading = 2*np.pi

    C = PVelocityControl(M, v_max=10, w_max=90/180*np.pi)

    S = Simulation(E, C, goal=np.array([10,50]), render=True, window_size=(1050, 550), FPS=60, render_offset=Offset(50,0,0,0), center_col_width=50)

    S.render_frame()

    S.check_render_next()

    S.run_sim()

    S.check_end_sim()