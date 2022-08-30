import math
import torch
import random
import pygame as pg
import numpy as np
import constants as const

from collections import deque
from model import Linear_QNet, QTrainer
from helper import distance

vec = pg.math.Vector2

MAX_MEMORY = 100_000 # Storing 100k max items in deque
BATCH_SIZE = 1000
N_INPUTS = 14
N_HIDDEN = 256
N_OUTPUTS = 4
LR = 0.001

class Agent(pg.sprite.Sprite):
    # -----------
    def __init__(self, game):
        pg.sprite.Sprite.__init__(self)
        self.game = game
        self.size = const.BLOCK_SIZE
        self.place()
        self.vel = vec(0, 0)
        self.direction = None
        self.reset_ok = False
        self.color = pg.Color("Blue")
        
        # DQN
        self.n_games = 0
        self.n_exploration = 0
        self.n_exploitation = 0
        self.epsilon = 0.1
        self.max_epsilon = self.epsilon
        self.min_epsilon = 0.001
        self.decay = 0.01
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(input_size=N_INPUTS, hidden_size=N_HIDDEN, output_size=N_OUTPUTS)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma) 

    def place(self):
        x = 13*const.BLOCK_SIZE - self.size//2
        y = 8*const.BLOCK_SIZE - self.size//2
        
        self.pos = vec(x, y)
        self.rect = pg.Rect(self.pos.x, self.pos.y, self.size, self.size)
        
    def closest_enemy(self):
        distances_enemies = [distance(self.pos, enemy.pos) for enemy in self.game.enemies]
        return min(distances_enemies), distances_enemies.index(min(distances_enemies))

    def wall_collision(self, offset):
        return (
            self.rect.left - offset < 0 or 
            self.rect.right + offset > const.WIDTH or
            self.rect.top - offset < 0 or
            self.rect.bottom + offset > const.HEIGHT
        )

    def enemy_collision(self):
        # If the agent is already colliding with an enemy, returns True directly
        if pg.sprite.spritecollide(self, self.game.enemies, False):
            return True
        
    def enemy_danger(self, danger_direction):
        buffer_rect = self.rect.copy()
        
        if danger_direction == "left":
            buffer_rect.left -= const.AGENT_X_SPEED
        elif danger_direction == "right":
            buffer_rect.right += const.AGENT_X_SPEED
        elif danger_direction == "down":
            buffer_rect.bottom += const.AGENT_X_SPEED
        elif danger_direction == "up":
            buffer_rect.top -= const.AGENT_Y_SPEED

        for enemy in self.game.enemies:
            if buffer_rect.colliderect(enemy.rect):
                return True    
        return False
    
    def enemy_collision(self):
        collisions_sprites = pg.sprite.spritecollide(self, self.game.enemies, False)
        
        if collisions_sprites:
            for sprite in collisions_sprites:
                if self.direction == "right" and self.rect.right >= sprite.rect.left:
                    self.rect.right = sprite.rect.left
                    self.pos.x = self.rect.centerx
                    break
                # Left 
                if self.direction == "left" and self.rect.left <= sprite.rect.right:
                    self.rect.left = sprite.rect.right
                    self.pos.x = self.rect.centerx
                    break
                # Bottom 
                if self.direction == "up" and self.rect.bottom >= sprite.rect.top:
                    self.rect.bottom = sprite.rect.top
                    self.pos.y = self.rect.centery
                    break
                # Top
                if self.direction == "down" and self.rect.top <= sprite.rect.bottom:
                    self.rect.top = sprite.rect.bottom
                    self.pos.y = self.rect.centery
                    break
        
        return bool(collisions_sprites)                    
    
    def food_collision(self):
        return self.rect.colliderect(self.game.food.rect)
    
    def get_state(self) -> np.array:     
        return np.array([
            # current direction
            self.direction == "stand",
            self.direction == "right",
            self.direction == "left",
            self.direction == "down",
            self.direction == "up",
     
            # food location
            self.rect.right <= self.game.food.rect.left, # food is right
            self.rect.left >= self.game.food.rect.right, # food is left
            self.rect.bottom <= self.game.food.rect.top, # food is bottom
            self.rect.top >= self.game.food.rect.bottom, # food is up
            
            # dangers
            self.enemy_danger("right"),
            self.enemy_danger("left"),
            self.enemy_danger("down"),
            self.enemy_danger("up"),
            self.wall_collision(offset=const.BLOCK_SIZE)], dtype=int
        )    

    def get_action(self, state):
        final_move = [0]*N_OUTPUTS
        rand_number = np.random.rand()
        
        if rand_number <= self.epsilon:
            self.n_exploration += 1
            move = random.randint(0, N_OUTPUTS-1)
        else:
            self.n_exploitation += 1
            state_0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_0)
            move = torch.argmax(prediction).item() # returns index of max value
                
        final_move[move] = 1
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) \
                       * np.exp(-self.decay * self.n_games)
            
        return final_move

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            # returns list of tuples
            mini_sample = random.sample(self.memory, BATCH_SIZE) 
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def remember(self, state, action, reward, next_state, done):
        # will pop left if MAX_MEMORY is reached
        self.memory.append((state, action, reward, next_state, done))

    def update(self, action):        
        # Tests only
        if self.game.human:
            keys = pg.key.get_pressed() # Keyboard events
            if keys[pg.K_RIGHT]:
                self.direction = "right"
                self.pos.x += const.AGENT_X_SPEED
            elif keys[pg.K_LEFT]:
                self.direction = "left"
                self.pos.x += -const.AGENT_X_SPEED
            elif keys[pg.K_UP]:
                self.direction = "down"
                self.pos.y += -const.AGENT_Y_SPEED
            elif keys[pg.K_DOWN]:
                self.direction = "up"
                self.pos.y += const.AGENT_Y_SPEED
        else:
            if np.array_equal(action, [1, 0, 0, 0]): # going right
                self.direction = "right"
                self.pos.x += const.AGENT_X_SPEED
            elif np.array_equal(action, [0, 1, 0, 0]): # going left
                self.direction = "left"
                self.pos.x += -const.AGENT_X_SPEED
            elif np.array_equal(action, [0, 0, 1, 0]): # going down
                self.direction = "down"
                self.pos.y += -const.AGENT_Y_SPEED
            elif np.array_equal(action, [0, 0, 0, 1]): # going up
                self.direction = "up"
                self.pos.y += const.AGENT_Y_SPEED
                
            # if np.array_equal(action, [1, 0, 0, 0, 0]): # standing still
            #     self.direction = "stand"
            # if np.array_equal(action, [0, 1, 0, 0, 0]): # going right
            #     self.direction = "right"
            #     self.pos.x += const.AGENT_X_SPEED
            # elif np.array_equal(action, [0, 0, 1, 0, 0]): # going left
            #     self.direction = "left"
            #     self.pos.x += -const.AGENT_X_SPEED
            # elif np.array_equal(action, [0, 0, 0, 1, 0]): # going down
            #     self.direction = "down"
            #     self.pos.y += -const.AGENT_Y_SPEED
            # elif np.array_equal(action, [0, 0, 0, 0, 1]): # going up
            #     self.direction = "up"
            #     self.pos.y += const.AGENT_Y_SPEED
            
        # Updating pos
        # self.pos += self.vel
        self.rect.center = self.pos

if __name__ == "__main__":
    pass




