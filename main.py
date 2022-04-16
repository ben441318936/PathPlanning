import sys, pygame
import numpy as np

pygame.init()

clock = pygame.time.Clock()

size = width, height = 500, 400
speed = np.array([2, 2])
black = 0, 0, 0

screen = pygame.display.set_mode(size)

ball_pos = np.array([50,50])
ball_radius = 10

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()

    screen.fill(black)

    ball = pygame.draw.circle(screen, "red", ball_pos, ball_radius)

    if ball_pos[0] - ball_radius < 0 or ball_pos[0] + ball_radius > width:
        speed[0] = -speed[0]
    if ball_pos[1] - ball_radius < 0 or ball_pos[1] + ball_radius > height:
        speed[1] = -speed[1]

    ball_pos += speed

    pygame.display.flip()
    clock.tick(60)

