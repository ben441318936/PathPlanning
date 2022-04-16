import pygame
import sys
import numpy as np

pygame.init()

class Renderer(object):
    def __init__(self, window_size, FPS) -> None:
        self.window_size = window_size
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode(window_size)

        self.FPS = FPS

        self.speed = np.array([2, 2])
        self.ball_pos = np.array([50,50])
        self.ball_radius = 10

        self.color_dict = {}
        self.color_dict["black"] = (0,0,0)
        self.color_dict["red"] = (255,0,0)

    def draw_frame(self) -> None:
        self.screen.fill(self.color_dict["black"])

        ball = pygame.draw.circle(self.screen, self.color_dict["red"], self.ball_pos, self.ball_radius)

        if self.ball_pos[0] - self.ball_radius < 0 or self.ball_pos[0] + self.ball_radius > self.window_size[0]:
            self.speed[0] *= -1
        if self.ball_pos[1] - self.ball_radius < 0 or self.ball_pos[1] + self.ball_radius > self.window_size[1]:
            self.speed[1] *= -1

        self.ball_pos += self.speed

    def render_frame(self) -> None:
        '''
        Renders the frame buffer.
        Includes the maximum refresh rate delay.
        '''
        pygame.display.flip()
        self.clock.tick(self.FPS)

    def start_render_loop(self) -> None:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit()
            self.draw_frame()
            self.render_frame()


if __name__ == "__main__":
    R = Renderer((500, 400), 60)
    R.start_render_loop()

    