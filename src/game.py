import pygame
import random


NORTH = 1
EAST = 2
SOUTH = 3
WEST = 4
cols, rows = (15, 15)
block_size = 30

snake = [(0, 2), (0, 1), (0, 0)]
prev_direction = SOUTH
direction = SOUTH
food = (random.randint(0, cols-1), random.randint(0, rows-1))
while food in snake:
    food = (random.randint(0, cols-1), random.randint(0, rows-1))
score = 0
snake_grow = False

pygame.init()
pygame.display.set_caption('Snake (Score: 0)')
screen = pygame.display.set_mode(((block_size + 2) * cols, (block_size + 2) * rows))

def is_dead():
    #Left, Right, Top, Bottom collision
    if snake[0][0] < 0 or snake[0][0] > cols-1 or snake[0][1] < 0 or snake[0][1] > rows-1:
        return True
	
    #Body collision
    for i in range(1, len(snake) - 1):
        if (snake[0][0] == snake[i][0]) and (snake[0][1] == snake[i][1]):
            return True
    return False

def draw_rect(color, row, col):
    pygame.draw.rect(screen, color, (row*(block_size+2)+1, col*(block_size+2)+1, block_size, block_size))
done = False
while not done:
    pygame.time.delay(100)
    pygame.display.set_caption("Snake (Score: " + str(score) + ")")
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP and prev_direction != SOUTH:
                direction = NORTH
            elif event.key == pygame.K_DOWN and prev_direction != NORTH:
                direction = SOUTH
            elif event.key == pygame.K_LEFT and prev_direction != EAST:
                direction = WEST
            elif event.key == pygame.K_RIGHT and prev_direction != WEST:
                direction = EAST

    screen.fill((30, 30, 30))
    pygame.display.flip()

    # Move snake body
    # Iterate through snake backwards, excluding the head
    if snake_grow:
        snake.append((-1, -1))
        snake_grow = False
    for i in range(len(snake) - 1, 0, -1):
        snake[i] = snake[i - 1]
    prev_direction = direction

    # Move snake head
    head = snake[0]
    if direction == NORTH:
        snake[0] = (head[0], head[1] - 1)
    elif direction == SOUTH:
        snake[0] = (head[0], head[1] + 1)
    elif direction == WEST:
        snake[0] = (head[0] - 1, head[1])
    elif direction == EAST:
        snake[0] = (head[0] + 1, head[1])
    
    #Check death collisions
    if is_dead():
        done = True

    #Check food collision
    if snake[0][0] == food[0] and snake[0][1] == food[1]:
        score += 1
        food = (random.randint(0, cols-1), random.randint(0, rows-1))
        while food in snake:
            food = (random.randint(0, cols-1), random.randint(0, rows-1))
        snake_grow = True        
    #Iterate for Empty Set then Random on Set
    #Make ALl points set randomize then check (Remove if part of snake)
    #Make ALL points set then iterate through snake and remove points on snake

    # Draw the board
    for r in range(rows):
        for c in range(cols):
            draw_rect((10, 10, 10), r, c)
    # Draw the Snake over board
    for i in range(len(snake)):
        draw_rect((0, 255, 0), snake[i][0], snake[i][1])
    # Draw the food over board
        draw_rect((255, 0, 0), food[0], food[1])

    # So we don't show snake out of board
    if not done:
        pygame.display.update()

pygame.quit()