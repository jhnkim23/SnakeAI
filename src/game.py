import pygame
import random


NORTH = 1
EAST = 2
SOUTH = 3
WEST = 4
cols, rows = (10, 10)
block_width, block_height = (30, 30)

snake = [(0, 2), (0, 1), (0, 0)]
prev_direction = SOUTH
direction = SOUTH
food = (random.randint(0, cols-1), random.randint(0, rows-1))
score = 0
snake_grow = False

pygame.init()
pygame.display.set_caption('Snake (Score: 0)')
screen = pygame.display.set_mode((block_width * cols + 27, block_height * rows + 27))

def isDead(snake):
    #Left, Right, Top, Bottom collision
    if snake[0][0] < 0 or snake[0][0] > cols-1 or snake[0][1] < 0 or snake[0][1] > rows-1:
        return True
	
    #Body collision
    for i in range(1, len(snake) - 1):
        if (snake[0][0] == snake[i][0]) and (snake[0][1] == snake[i][1]):
            return True
    return False

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

        screen.fill((0, 0, 0))
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
    if isDead(snake):
        done = True

    #Check food collision
    if snake[0][0] == food[0] and snake[0][1] == food[1]:
        score += 1
        food = (random.randint(0, cols-1), random.randint(0, rows-1))
        snake_grow = True
    
    #!!!There r prob gonna b some alignment errors here!!!
    # Draw the Board
    for r in range(rows):
        for c in range(cols):
            pygame.draw.rect(screen, (255, 255, 255), (r*33, c*33, 30, 30))
    # Draw the Snake over Board
    for i in range(len(snake)):
        pygame.draw.rect(screen, (0, 255, 0),
                         (snake[i][0]*33, snake[i][1]*33, 30, 30))
    pygame.draw.rect(screen, (255, 0, 0), (food[0] * 33, food[1]*33,30,30))

    # So we don't show snake out of board
    if not done:
        pygame.display.update()

pygame.quit()
