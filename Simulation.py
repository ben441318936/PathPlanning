from hypothesis import target
import numpy as np
import sys
from Agent import Agent
from Grid import Grid

class Simulation(object):
    def __init__(self, window_size=None, FPS=None, render=False) -> None:
        self.agent = None
        self.grid = None
        self.render = render
        if render and (window_size is None or FPS is None):
            print("Render set to True, but no render parameters given.")
            sys.exit()
        if render:
            import pygame
            pygame.init()
        self.window_size = window_size
        self.FPS = FPS

    def init_grid(self, grid_size, agent_pos, target_pos) -> None:
        self.grid = Grid(grid_size)
        if not (self.grid.place_target(target_pos) and self.grid.place_agent(agent_pos)):
            print("Grid initialization failed with grid_size={}, agent_pos={}, target_pos={}".format(grid_size,agent_pos,target_pos))
            sys.exit()

    def init_agent(self, init_map_size, max_map_size) -> None:
        self.agent = Agent(init_map_size, max_map_size)

    def reset(self) -> None:
        self.grid = None
        self.agent = None

    def run_sim(self) -> None:
        # render set up
        if self.render:
            pass

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

        if not finished:
            print("Sim loop exited without agent reaching target.")


if __name__ == "__main__":
    sim = Simulation()
    sim.init_grid((10,10), (8,8), (0,4))
    sim.init_agent((5,5), (30,30))

    # sim.grid.set_obstacle((5,slice(0,9,None)))
    sim.grid.set_random_obstacle(0.2)
    
    print("Initial grid:")
    sim.grid.print_grid()

    print("Initial map:")
    sim.agent.print_map()

    sim.run_sim()

    print("Final grid:")
    sim.grid.print_grid()

    print("Final map:")
    sim.agent.print_map()