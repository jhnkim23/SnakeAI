import pygame
import random
import math

NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3
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

def is_dead(snake):
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
def step(action):
    #pygame.time.delay(100)
    pygame.display.set_caption("Snake (Score: " + str(score) + ")")
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
    #     if event.type == pygame.KEYDOWN:
    #         if event.key == pygame.K_UP and prev_direction != SOUTH:
    #             direction = NORTH
    #         elif event.key == pygame.K_DOWN and prev_direction != NORTH:
    #             direction = SOUTH
    #         elif event.key == pygame.K_LEFT and prev_direction != EAST:
    #             direction = WEST
    #         elif event.key == pygame.K_RIGHT and prev_direction != WEST:
    #             direction = EAST
    screen.fill((30, 30, 30))
    pygame.display.flip()

    #idk if this is in the right spot
    direction = (prev_direction+action)%4

    # Initialize state output array
    state = [0] * 8

    #Calculate initial distance from snake head to food
    initial_dist_to_food = math.sqrt((food[0] - snake[0][0])**2 + (food[1] - snake[0][1])**2)
    
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
    
    # Calculate final distance from snake head to food
    post_dist_to_food = math.sqrt((food[0] - snake[0][0])**2 + (food[1] - snake[0][1])**2)
    if post_dist_to_food < initial_dist_to_food:
        state[6] = 1

    #Check death collisions
    if is_dead(snake):
        done = True

    #Check food collision
    if snake[0][0] == food[0] and snake[0][1] == food[1]:
        state[7] = 1
        score += 1
        food = (random.randint(0, cols-1), random.randint(0, rows-1))
        while food in snake:
            food = (random.randint(0, cols-1), random.randint(0, rows-1))
        snake_grow = True        

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
    # if not done
    pygame.display.update()
    
    # Return state array for reinforcement learning:
        # Whether empty: Left, front, right
        # Whether food is in that direction: Left, front, right
        # Whether getting closer to food from last frame
        # Whether eating food in current frame

    #Check first 6 states
    if direction == NORTH:
        #Check if left is empty
        snake_copy = snake.copy()
        snake_copy[0][0] -= 1
        state[0] = 0 if is_dead(snake_copy) else 1
                
        #Check is front is empty
        snake_copy = snake.copy()
        snake_copy[0][1] -= 1
        state[1] = 0 if is_dead(snake_copy) else 1
        
        #Check is right is empty
        snake_copy = snake.copy()
        snake_copy[0][0] += 1
        state[2] = 0 if is_dead(snake_copy) else 1

        #Check direction of food relative to snake head
        #Food is towards the left
        if food[0] < snake[0][0]:
            state[3] = 1
        #Food is in front
        if food[1] < snake[0][1]:
            state[4] = 1
        #Food is toward the right
        if food[0] > snake[0][0]:
            state[5] = 1

    elif direction == SOUTH:
        #Check if left is empty
        snake_copy = snake.copy()
        snake_copy[0][0] += 1
        state[0] = 0 if is_dead(snake_copy) else 1
                
        #Check is front is empty
        snake_copy = snake.copy()
        snake_copy[0][1] += 1
        state[1] = 0 if is_dead(snake_copy) else 1
        
        #Check if right is empty
        snake_copy = snake.copy()
        snake_copy[0][0] -= 1
        state[2] = 0 if is_dead(snake_copy) else 1

        #Check direction of food relative to snake head
        #Food is towards the left
        if food[0] > snake[0][0]:
            state[3] = 1
        #Food is in front
        if food[1] > snake[0][1]:
            state[4] = 1
        #Food is toward the right
        if food[0] < snake[0][0]:
            state[5] = 1
        
    elif direction == WEST:
        #Check is left is empty
        snake_copy = snake.copy()
        snake_copy[0][1] += 1
        state[0] = 0 if is_dead(snake_copy) else 1
                
        #Check if front is empty
        snake_copy = snake.copy()
        snake_copy[0][0] -= 1
        state[1] = 0 if is_dead(snake_copy) else 1
        
        #Check if right is empty
        snake_copy = snake.copy()
        snake_copy[0][1] -= 1
        state[2] = 0 if is_dead(snake_copy) else 1
        
        #Check direction of food relative to snake head
        #Food is towards the left
        if food[1] > snake[0][1]:
            state[3] = 1
        #Food is in front
        if food[0] < snake[0][0]:
            state[4] = 1
        #Food is toward the right
        if food[1] < snake[0][1]:
            state[5] = 1

    else: # EAST
        #Check if left is empty
        snake_copy = snake.copy()
        snake_copy[0][1] -= 1
        state[0] = 0 if is_dead(snake_copy) else 1

        #Check if front is empty
        snake_copy = snake.copy()
        snake_copy[0][0] += 1
        state[1] = 0 if is_dead(snake_copy) else 1

        #Check is right is empty
        snake_copy = snake.copy()
        snake_copy[0][1] += 1
        state[2] = 0 if is_dead(snake_copy) else 1

        #Check direction of food relative to snake head
        #Food is towards the left
        if food[1] < snake[0][1]:
            state[3] = 1
        #Food is in front
        if food[0] > snake[0][0]:
            state[4] = 1
        #Food is toward the right
        if food[1] > snake[0][1]:
            state[5] = 1
    
    return state

pygame.quit()
