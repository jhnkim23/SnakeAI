import pygame
import random
import math

class Snake:
    def __init__(self):
        self.NORTH = 0
        self.EAST = 1
        self.SOUTH = 2
        self.WEST = 3
        self.cols, self.rows = (15, 15)
        self.block_size = 30

        self.snake = [(1, 3), (1, 2), (1, 1)]
        self.prev_direction = self.SOUTH
        self.direction = self.SOUTH
        self.food = (random.randint(0, self.cols-1), random.randint(0, self.rows-1))
        while self.food in self.snake:
            self.food = (random.randint(0, self.cols-1), random.randint(0, self.rows-1))
        self.score = 0
        self.done = False #NOT USED IN THIS FILE
        self.snake_grow = False
        pygame.init()
        pygame.display.set_caption('Snake (Score: 0)')
        self.screen = pygame.display.set_mode(((self.block_size + 2) * self.cols, (self.block_size + 2) * self.rows))

    def is_dead(self, snake):
        #Left, Right, Top, Bottom collision
        if snake[0][0] < 0 or snake[0][0] > self.cols-1 or snake[0][1] < 0 or snake[0][1] > self.rows-1:
            return True
        
        #Body collision
        for i in range(1, len(snake) - 1):
            if (snake[0][0] == snake[i][0]) and (snake[0][1] == snake[i][1]):
                return True
        return False

    def draw_rect(self, color, row, col):
        pygame.draw.rect(self.screen, color, (row*(self.block_size+2)+1, col*(self.block_size+2)+1, self.block_size, self.block_size))

    def step(self, action):
        #pygame.time.delay(100)
        pygame.display.set_caption("Snake (Score: " + str(self.score) + ")")
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True
            # if event.type == pygame.KEYDOWN:
            #     if event.key == pygame.K_UP and self.prev_direction != self.SOUTH:
            #         self.direction = self.NORTH
            #     elif event.key == pygame.K_DOWN and self.prev_direction != self.NORTH:
            #         self.direction = self.SOUTH
            #     elif event.key == pygame.K_LEFT and self.prev_direction != self.EAST:
            #         self.direction = self.WEST
            #     elif event.key == pygame.K_RIGHT and self.prev_direction != self.WEST:
            #         self.direction = self.EAST
        self.screen.fill((30, 30, 30))
        pygame.display.flip()

        #idk if this is in the right spot
        self.direction = (self.prev_direction+action)%4

        # Initialize state output array
        state = [0] * 8

        #Calculate initial distance from snake head to food
        initial_dist_to_food = math.sqrt((self.food[0] - self.snake[0][0])**2 + (self.food[1] - self.snake[0][1])**2)
        
        # Move snake body
        # Iterate through snake backwards, excluding the head
        if self.snake_grow:
            self.snake.append((-1, -1))
            self.snake_grow = False
        for i in range(len(self.snake) - 1, 0, -1):
            self.snake[i] = self.snake[i - 1]
        self.prev_direction = self.direction

        # Move snake head
        head = self.snake[0]
        if self.direction == self.NORTH:
            self.snake[0] = (head[0], head[1] - 1)
        elif self.direction == self.SOUTH:
            self.snake[0] = (head[0], head[1] + 1)
        elif self.direction == self.WEST:
            self.snake[0] = (head[0] - 1, head[1])
        elif self.direction == self.EAST:
            self.snake[0] = (head[0] + 1, head[1])
        
        # Calculate final distance from snake head to food
        post_dist_to_food = math.sqrt((self.food[0] - self.snake[0][0])**2 + (self.food[1] - self.snake[0][1])**2)
        if post_dist_to_food < initial_dist_to_food:
            state[6] = 1

        #Check death collisions
        if self.is_dead(self.snake):
            self.done = True

        #Check self.food collision
        if self.snake[0][0] == self.food[0] and self.snake[0][1] == self.food[1]:
            state[7] = 1
            self.score += 1
            self.food = (random.randint(0, self.cols-1), random.randint(0, self.rows-1))
            while self.food in self.snake:
                self.food = (random.randint(0, self.cols-1), random.randint(0, self.rows-1))
            self.snake_grow = True        

        # Draw the board
        for r in range(self.rows):
            for c in range(self.cols):
                self.draw_rect((10, 10, 10), r, c)
        # Draw the self.Snake over board
        for i in range(len(self.snake)):
            self.draw_rect((0, 255, 0), self.snake[i][0], self.snake[i][1])
        # Draw the self.food over board
            self.draw_rect((255, 0, 0), self.food[0], self.food[1])

        # So we don't show self.snake out of board
        # if not done
        pygame.display.update()
        
        # Return state array for reinforcement learning:
            # Whether empty: Left, front, right
            # Whether self.food is in that direction: Left, front, right
            # Whether getting closer to self.food from last frame
            # Whether eating self.food in current frame

        #Check first 6 states
        if self.direction == self.NORTH:
            #Check if left is empty
            self.snake_copy = self.snake.copy()
            self.snake_copy[0] = (self.snake_copy[0][0] - 1, self.snake_copy[0][1])
            state[0] = 0 if self.is_dead(self.snake_copy) else 1
                    
            #Check is front is empty
            self.snake_copy = self.snake.copy()
            self.snake_copy[0] = (self.snake_copy[0][0], self.snake_copy[0][1] - 1)
            state[1] = 0 if self.is_dead(self.snake_copy) else 1
            
            #Check is right is empty
            self.snake_copy = self.snake.copy()
            self.snake_copy[0] = (self.snake_copy[0][0] + 1, self.snake_copy[0][1])
            state[2] = 0 if self.is_dead(self.snake_copy) else 1

            #Check direction of self.food relative to self.snake head
            #self.Food is towards the left
            if self.food[0] < self.snake[0][0]:
                state[3] = 1
            #self.Food is in front
            if self.food[1] < self.snake[0][1]:
                state[4] = 1
            #self.Food is toward the right
            if self.food[0] > self.snake[0][0]:
                state[5] = 1

        elif self.direction == self.SOUTH:
            #Check if left is empty
            self.snake_copy = self.snake.copy()
            self.snake_copy[0] = (self.snake_copy[0][0] + 1, self.snake_copy[0][1])
            state[0] = 0 if self.is_dead(self.snake_copy) else 1
                    
            #Check is front is empty
            self.snake_copy = self.snake.copy()
            self.snake_copy[0] = (self.snake_copy[0][0], self.snake_copy[0][1] + 1)
            state[1] = 0 if self.is_dead(self.snake_copy) else 1
            
            #Check if right is empty
            self.snake_copy = self.snake.copy()
            self.snake_copy[0] = (self.snake_copy[0][0] - 1, self.snake_copy[0][1])
            state[2] = 0 if self.is_dead(self.snake_copy) else 1

            #Check direction of self.food relative to self.snake head
            #self.Food is towards the left
            if self.food[0] > self.snake[0][0]:
                state[3] = 1
            #self.Food is in front
            if self.food[1] > self.snake[0][1]:
                state[4] = 1
            #self.Food is toward the right
            if self.food[0] < self.snake[0][0]:
                state[5] = 1
            
        elif self.direction == self.WEST:
            #Check is left is empty
            self.snake_copy = self.snake.copy()
            self.snake_copy[0] = (self.snake_copy[0][0], self.snake_copy[0][1] + 1)
            state[0] = 0 if self.is_dead(self.snake_copy) else 1
                    
            #Check if front is empty
            self.snake_copy = self.snake.copy()
            self.snake_copy[0] = (self.snake_copy[0][0] - 1, self.snake_copy[0][1])
            state[1] = 0 if self.is_dead(self.snake_copy) else 1
            
            #Check if right is empty
            self.snake_copy = self.snake.copy()
            self.snake_copy[0] = (self.snake_copy[0][0], self.snake_copy[0][1] - 1)
            state[2] = 0 if self.is_dead(self.snake_copy) else 1
            
            #Check direction of self.food relative to self.snake head
            #self.Food is towards the left
            if self.food[1] > self.snake[0][1]:
                state[3] = 1
            #self.Food is in front
            if self.food[0] < self.snake[0][0]:
                state[4] = 1
            #self.Food is toward the right
            if self.food[1] < self.snake[0][1]:
                state[5] = 1

        else: # self.EAST
            #Check if left is empty
            self.snake_copy = self.snake.copy()
            self.snake_copy[0] = (self.snake_copy[0][0], self.snake_copy[0][1] - 1)
            state[0] = 0 if self.is_dead(self.snake_copy) else 1

            #Check if front is empty
            self.snake_copy = self.snake.copy()
            self.snake_copy[0] = (self.snake_copy[0][0] + 1, self.snake_copy[0][1])
            state[1] = 0 if self.is_dead(self.snake_copy) else 1

            #Check is right is empty
            self.snake_copy = self.snake.copy()
            self.snake_copy[0] = (self.snake_copy[0][0], self.snake_copy[0][1] + 1)
            state[2] = 0 if self.is_dead(self.snake_copy) else 1

            #Check direction of self.food relative to snake head
            #self.Food is towards the left
            if self.food[1] < self.snake[0][1]:
                state[3] = 1
            #self.Food is in front
            if self.food[0] > self.snake[0][0]:
                state[4] = 1
            #self.Food is toward the right
            if self.food[1] > self.snake[0][1]:
                state[5] = 1
        
        return state
    
    def return_state(self):
        return_state = [0] * 8
        return_state[0:3] = [1, 1, 1]

        #Check direction of self.food relative to self.snake head
        #self.Food is towards the left
        if self.food[0] > self.snake[0][0]:
            return_state[3] = 1
        #self.Food is in front
        if self.food[1] > self.snake[0][1]:
            return_state[4] = 1
        #self.Food is toward the right
        if self.food[0] < self.snake[0][0]:
            return_state[5] = 1
        
        return return_state
    #pygame.quit()