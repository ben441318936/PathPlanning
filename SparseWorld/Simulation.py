from collections import namedtuple
import numpy as np
np.set_printoptions(precision=2)
import sys

from time import time

from functools import partial

import pygame
pygame.init()

Offset = namedtuple("Offset", ["top", "bottom", "left", "right"])

class Simulation(object):
    def __init__(self, render=False, window_size=None, FPS=None, render_offset=(0,0,0,0), center_col_width=0) -> None:
        self.render = render
        self.window_size = window_size
        self.FPS = FPS
        self.render_offset = render_offset # (top,bottom,left,right)
        self.center_col_width = center_col_width

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

    def run_sim(self, manual=False) -> None:
        pass

    def draw_env(self, env_rect) -> None:
        # draw the obstacles and agents, scaled to fit in env_rect
        pass
        # draw obstacles

        # draw agent


        

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
    pass