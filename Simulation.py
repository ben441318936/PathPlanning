from collections import namedtuple
import numpy as np
import sys

import pygame
from Agent import A_star_agent, D_star_agent, MapStatus
from Grid import Grid, GridStatus

Offset = namedtuple("Offset", ["top", "bottom", "left", "right"])

class Simulation(object):
    def __init__(self, render=False, window_size=None, FPS=None, render_offset=(0,0,0,0), center_col_width=0) -> None:
        self.agent = None
        self.grid = None
        self.render = render
        self.window_size = window_size
        self.FPS = FPS
        self.render_offset = render_offset # (top,bottom,left,right)
        self.center_col_width = center_col_width

        if render and (window_size is None or FPS is None):
            print("Render set to True, but no render parameters given.")
            sys.exit()
        if render:
            import pygame
            pygame.init()
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

    def init_grid(self, grid_size) -> None:
        self.grid = Grid(grid_size)

    def fill_random_grid(self, agent_pos=None, target_pos=None, probability=0.3, seed=None) -> None:
        np.random.seed(seed)
        self.grid.fill_random_grid(probability)
        if agent_pos is None:
            self.grid.set_random_agent()
        else:
            self.grid.place_agent(agent_pos, force=True)
        if target_pos is None:
            self.grid.set_random_target()
        else:
            self.grid.place_target(target_pos, force=True)

    def init_agent(self, agent_class) -> None:
        if self.grid is None:
            print("No grid, can't initialize agent.")
            sys.exit()
        self.agent = agent_class()
        self.agent.set_target(self.grid.relative_target_pos())

    def reset(self) -> None:
        self.grid = None
        self.agent = None

    def run_sim(self) -> None:
        if self.render:
            self.render_frame()

        # # do an initial scan
        # self.agent.update_map(self.grid.scan(self.agent.cone_of_vision()))

        finished = self.grid.agent_reached_target()
        # search loop
        while not finished: 
            # check for possible mismatches between agent map and grid
            if self.agent.reached_target() != self.grid.agent_reached_target():
                print("Mismatch between agent knowledge and grid status.")
                break
            # check if agent path is still valid
            if not self.agent.path_valid():
                # try to plan a path
                if not self.agent.plan():
                    print("Planning failed.")
                    break
            # try to take next step in path
            if not self.grid.agent_move(self.agent.next_action()):
                print("Environment does not allow the next action.")
                break
            # if environment allows this action
            if not self.agent.take_next_action():
                print("Agent knowledge does not allow the next action.")
                break
            # agent scan and update
            self.agent.update_map(self.grid.scan(self.agent.cone_of_vision()))
            # check for reaching target
            finished = self.grid.agent_reached_target()

            # render
            if self.render:
                self.render_frame()

        if not finished:
            print("Sim loop exited without agent reaching target.")

    def draw_grid(self, grid_rect) -> None:
        grid_size = self.grid.size()

        # compute grid cell sizing
        # in pygame, first coordinate is horizontal
        horizontal_width = grid_rect.width // grid_size[1]
        vertical_width = grid_rect.height // grid_size[0]

        cell_width = min(horizontal_width, vertical_width)

        curr_corner = np.array([grid_rect.left, grid_rect.top])

        cells = [] # used to store all the drawn rectangles

        for i in range(grid_size[0]):
            cells.append([])
            for j in range(grid_size[1]):
                # set cell fill color base on cell status
                cell_status = self.grid.get_cell((i,j))
                if cell_status == GridStatus.AGENT:
                    c = self.color_dict["blue"]
                elif cell_status == GridStatus.PREV_AGENT:
                    c = self.color_dict["turquoise"]
                elif cell_status == GridStatus.TARGET:
                    c = self.color_dict["green"]
                elif cell_status == GridStatus.BOTH:
                    c = self.color_dict["yellow"]
                elif cell_status == GridStatus.OBSTACLE:
                    c = self.color_dict["brown"]
                else:
                    c = self.color_dict["black"]
                # set location
                rect = pygame.Rect(curr_corner[0], curr_corner[1], cell_width, cell_width)
                cells[-1].append(rect)
                # draw the cell
                pygame.draw.rect(self.screen, c, rect)
                # draw cell border
                pygame.draw.rect(self.screen, self.color_dict["gray"], rect, 1)
                curr_corner += np.array([cell_width,0])
            curr_corner[0] = grid_rect.left
            curr_corner += np.array([0,cell_width])

        # draw the scan border
        corner = np.array([grid_rect.left, grid_rect.top])
        corner += np.array([self.grid.agent_pos[1]-1, self.grid.agent_pos[0]-1]) * cell_width
        scan_width = 3 * cell_width
        pygame.draw.rect(self.screen, self.color_dict["red"], pygame.Rect(corner[0], corner[1], scan_width, scan_width), 2)

        # draw the agent's planned path
        path = self.agent.get_path_agent_frame()
        if path.shape[0] != 0:
            path = self.grid.translate_path_to_world_frame(path)
            for k in range(path.shape[0]-1):
                curr_coord = path[k]
                next_coord = path[k+1]
                if self.grid.in_bounds(curr_coord) and self.grid.in_bounds(next_coord):
                    pygame.draw.line(self.screen, self.color_dict["purple"], 
                        cells[curr_coord[0]][curr_coord[1]].center, cells[next_coord[0]][next_coord[1]].center, 3)
                else:
                    break

    def draw_map(self, map_rect) -> None:
        map_size = self.agent.map_size()

        # compute grid cell sizing
        # in pygame, first coordinate is horizontal
        horizontal_width = map_rect.width // map_size[1]
        vertical_width = map_rect.height // map_size[0]

        cell_width = min(horizontal_width, vertical_width)

        curr_corner = np.array([map_rect.left, map_rect.top])

        cells = [] # used to store all the drawn rectangles

        for i in range(map_size[0]):
            cells.append([])
            for j in range(map_size[1]):
                # set cell fill color base on cell status
                cell_status = self.agent.get_cell((i,j))
                if cell_status == MapStatus.AGENT:
                    c = self.color_dict["blue"]
                elif cell_status == MapStatus.TARGET:
                    c = self.color_dict["green"]
                elif cell_status == MapStatus.BOTH:
                    c = self.color_dict["yellow"]
                elif cell_status == MapStatus.OBSTACLE:
                    c = self.color_dict["brown"]
                else:
                    c = self.color_dict["black"]
                # set location
                rect = pygame.Rect(curr_corner[0], curr_corner[1], cell_width, cell_width)
                cells[-1].append(rect)
                # draw the cell
                pygame.draw.rect(self.screen, c, rect)
                # draw cell border
                pygame.draw.rect(self.screen, self.color_dict["gray"], rect, 1)
                curr_corner += np.array([cell_width,0])
            curr_corner[0] = map_rect.left
            curr_corner += np.array([0,cell_width])

        # draw the scan border
        corner = np.array([map_rect.left, map_rect.top])
        corner += np.array([self.agent.pos[1]-1, self.agent.pos[0]-1]) * cell_width
        scan_width = 3 * cell_width
        pygame.draw.rect(self.screen, self.color_dict["red"], pygame.Rect(corner[0], corner[1], scan_width, scan_width), 2)

        # draw the agent's planned path
        path = self.agent.get_path()
        if path.shape[0] != 0:
            for k in range(path.shape[0]-1):
                curr_coord = path[k]
                next_coord = path[k+1]
                if self.agent.in_bounds(curr_coord) and self.agent.in_bounds(next_coord):
                    pygame.draw.line(self.screen, self.color_dict["purple"], 
                        cells[curr_coord[0]][curr_coord[1]].center, cells[next_coord[0]][next_coord[1]].center, 3)
                else:
                    break

    def render_frame(self) -> None:
        if not self.render:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()

        # clear window
        self.screen.fill(self.color_dict["black"])

        sub_figure_width = (self.window_size[0] - (self.render_offset.left + self.render_offset.right) - self.center_col_width) // 2
        sub_fugure_height = self.window_size[1] - (self.render_offset.top + self.render_offset.bottom)

        # draw the grid status
        grid_rect = pygame.Rect(self.render_offset.left, self.render_offset.top, 
                                sub_figure_width, 
                                sub_fugure_height)
        self.draw_grid(grid_rect)

        # grid status label text
        grid_label_rect = pygame.Rect(0, 0, (self.window_size[0]-self.center_col_width)//2, self.render_offset.top)
        self.screen.fill(self.color_dict["black"], grid_label_rect)
        my_font = pygame.font.SysFont("Times New Roman", 30)
        my_text = my_font.render("Grid Environment Status", True, self.color_dict["white"])
        my_rect = my_text.get_rect()
        width = my_rect.width
        height = my_rect.height
        self.screen.blit(my_text, (grid_label_rect.centerx - width//2, grid_label_rect.centery - height//2))

        # draw center column
        pygame.draw.rect(self.screen, self.color_dict["black"], 
                pygame.Rect((self.window_size[0]-self.center_col_width)//2, 0, self.center_col_width, self.window_size[1]))

        # draw the agent's map
        map_rect = pygame.Rect(sub_figure_width + self.center_col_width,
                                self.render_offset.top, 
                                sub_figure_width, 
                                sub_fugure_height)
        self.draw_map(map_rect)

        # map status label text
        map_label_rect = pygame.Rect((self.window_size[0]-self.center_col_width)//2 + self.center_col_width, 0,
                                        (self.window_size[0]-self.center_col_width)//2, self.render_offset.top)
        self.screen.fill(self.color_dict["black"], map_label_rect)
        my_font = pygame.font.SysFont("Times New Roman", 30)
        my_text = my_font.render("Agent Map", True, self.color_dict["white"])
        my_rect = my_text.get_rect()
        width = my_rect.width
        height = my_rect.height
        self.screen.blit(my_text, (map_label_rect.centerx - width//2, map_label_rect.centery - height//2))

        pygame.display.flip()
        self.clock.tick(self.FPS)


if __name__ == "__main__":
    sim = Simulation(render=True, window_size=(1050, 550), FPS=60, render_offset=Offset(50,0,0,0), center_col_width=50)
    map_width = map_height = 20
    sim.init_grid((map_height, map_width))
    sim.fill_random_grid(probability=0.4, seed=1)
    sim.init_agent(D_star_agent)

    sim.render_frame()

    started = False
    while not started:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN: 
                started = True

    sim.run_sim()

    sim.render_frame()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                sys.exit()
