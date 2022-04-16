from hypothesis import target
import numpy as np
import sys

import pygame
from Agent import Agent
from Grid import Grid, GridStatus

class Simulation(object):
    def __init__(self, render=False, window_size=None, FPS=None, render_offset=0) -> None:
        self.agent = None
        self.grid = None
        self.render = render
        self.window_size = window_size
        self.FPS = FPS
        self.render_offset = render_offset

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
                                "green": (0,255,0),
                                "yellow": (255,255,0),
                                "brown": (205,133,63),
                                "turquoise": (100,200,255),
                            }

    def init_grid(self, grid_size, agent_pos=None, target_pos=None) -> None:
        self.grid = Grid(grid_size)
        if agent_pos is None:
            self.grid.set_random_agent()
        else:
            self.grid.place_agent(agent_pos, force=True)
        if target_pos is None:
            self.grid.set_random_target()
        else:
            self.grid.place_target(target_pos, force=True)

    def init_agent(self, init_map_size, max_map_size) -> None:
        self.agent = Agent(init_map_size, max_map_size)

    def reset(self) -> None:
        self.grid = None
        self.agent = None

    def run_sim(self) -> None:
        if self.render:
            self.render_frame()

        self.agent.set_target(self.grid.relative_target_pos())
        # do an initial scan
        self.agent.update_map(self.grid.scan(self.agent.cone_of_vision()))

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
                    # if we can't find a path, expand the map and try again
                    # this assumes there are other empty spaces outside of current map scope
                    if self.agent.expand_map():
                        continue
                    else:
                        print("Reached max map size before finding a valid path.")
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

    def draw_map(self) -> None:
        self.screen.fill(self.color_dict["black"])

        grid_size = self.grid.size()

        # compute grid cell sizing
        # in pygame, first coordinate is horizontal
        horizontal_width = (self.window_size[0] - (2 * self.render_offset)) // grid_size[1]
        vertical_width = (self.window_size[1] - (2 * self.render_offset)) // grid_size[0]
        cell_width = min(horizontal_width, vertical_width)

        curr_corner = np.array([self.render_offset, self.render_offset])

        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                cell_status = self.grid.get_cell((i,j))
                if cell_status == GridStatus.AGENT:
                    c = self.color_dict["blue"]
                elif cell_status == GridStatus.PREV_AGENT:
                    c = self.color_dict["turquoise"]
                elif cell_status == GridStatus.TARGET:
                    c = self.color_dict["green"]
                elif cell_status == GridStatus.BOTH:
                    c = self.color_dict["yellow"]
                elif cell_status == GridStatus.WALL:
                    c = self.color_dict["brown"]
                else:
                    c = self.color_dict["black"]
                # draw the cell
                pygame.draw.rect(self.screen, c, pygame.Rect(curr_corner[0], curr_corner[1], cell_width, cell_width))
                # draw cell border
                pygame.draw.rect(self.screen, self.color_dict["gray"], pygame.Rect(curr_corner[0], curr_corner[1], cell_width, cell_width), 1)
                curr_corner += np.array([cell_width,0])
            curr_corner[0] = self.render_offset
            curr_corner += np.array([0,cell_width])

            # draw the scan border
            corner = np.array([self.render_offset, self.render_offset])
            corner += np.array([self.grid.agent_pos[1]-1, self.grid.agent_pos[0]-1]) * cell_width
            scan_width = 3 * cell_width
            pygame.draw.rect(self.screen, self.color_dict["red"], pygame.Rect(corner[0], corner[1], scan_width, scan_width), 2)

    def render_frame(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()
        self.draw_map()
        pygame.display.flip()
        self.clock.tick(self.FPS)


if __name__ == "__main__":
    sim = Simulation(render=True, window_size=(500, 500), FPS=5)
    map_size = 20
    sim.init_grid((map_size, map_size))
    sim.init_agent((5,5), (map_size*3, map_size*3))

    # sim.grid.set_obstacle((5,slice(0,9,None)))
    sim.grid.set_random_obstacle(0.3)
    
    # print("Initial grid:")
    # sim.grid.print_grid()

    # print("Initial map:")
    # sim.agent.print_map()

    sim.run_sim()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()

    # print("Final grid:")
    # sim.grid.print_grid()

    # print("Final map:")
    # sim.agent.print_map()