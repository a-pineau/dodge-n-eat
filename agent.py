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
N_INPUTS = 12
N_HIDDEN = 256
N_OUTPUTS = 4
LR = 0.001
X_AGENT, Y_AGENT = const.WIDTH*0.4, const.HEIGHT*0.5


class Agent(pg.sprite.Sprite):
    # -----------
    def __init__(self, game):
        pg.sprite.Sprite.__init__(self)
        self.game = game
        self.size = const.BLOCK_SIZE//2
        self.place()
        self.vel = vec(0, 0)
        self.color = pg.Color("Blue")
        # DQN
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(input_size=N_INPUTS, hidden_size=N_HIDDEN, output_size=N_OUTPUTS)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma) 

    def place(self):
        x, y = 4*const.BLOCK_SIZE - self.size, 4*const.BLOCK_SIZE - self.size
        self.pos = vec(x, y)
        self.rect = pg.Rect(self.pos.x, self.pos.y, self.size*2, self.size*2)
        self.rect.topleft = self.pos
        
    def closest_enemy(self):
        distances_enemies = [distance(self.pos, enemy.pos) for enemy in self.game.enemies]
        return min(distances_enemies), distances_enemies.index(min(distances_enemies))

    def wall_collision(self, offset):
        # left/right collision
        return (
            self.rect.left - const.BLOCK_SIZE < 0 or 
            self.rect.right + const.BLOCK_SIZE > const.WIDTH or
            self.rect.top - const.BLOCK_SIZE < 0 or
            self.rect.bottom + const.BLOCK_SIZE > const.HEIGHT
        )

    def enemy_collision(self, offset, direction=None):
        # If the agent is already colliding with an enemy, returns True directly
        if pg.sprite.spritecollide(self, self.game.enemies, False):
            return True
        # Otherwise we check in all 4 direction using an offset and a copy of the objet's rect
        buffer_rect = self.rect.copy()
        if direction == "left":
            buffer_rect.left -= offset
        elif direction == "right":
            buffer_rect.right += offset
        elif direction == "up":
            buffer_rect.top -= offset
        else:
            buffer_rect.bottom += offset
            
        for enemy in self.game.enemies:
            if buffer_rect.colliderect(enemy.rect):
                return True
        
        return False
            
    def food_collision(self):
        return self.rect.colliderect(self.game.food.rect)

    def get_state(self) -> np.array:
        # dangers
        walls_collision = self.wall_collision(offset=self.size) 
        # move directions
        going_right = self.vel.x > 0
        going_left = self.vel.x < 0
        going_up = self.vel.y < 0
        going_down = self.vel.y > 0
        # food location
        food_is_right = self.rect.right < self.game.food.rect.left 
        food_is_left = self.rect.left > self.game.food.rect.right
        food_is_up = self.rect.top > self.game.food.rect.bottom
        food_is_down = self.rect.bottom < self.game.food.rect.top

        states = [
            # going_right and self.enemy_collision(2*self.size, "right"),
            # going_left and self.enemy_collision(2*self.size, "left"),
            # going_down and self.enemy_collision(2*self.size, "down"),
            # going_up and self.enemy_collision(2*self.size, "up"),

            going_right and walls_collision,
            going_left and walls_collision,
            going_up and walls_collision,
            going_down and walls_collision,

            going_right,
            going_left,
            going_up,
            going_down,
            
            food_is_right,
            food_is_left,
            food_is_up,
            food_is_down,
        ]
        return np.array(states, dtype=int)

    def get_action(self, state):
        # trade-off exploration vs. exploitation
        self.epsilon = 100 - self.n_games
        final_move = [0, 0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, N_OUTPUTS-1)
        else:
            state_0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_0)
            move = torch.argmax(prediction).item() # returns index of max value
        final_move[move] = 1
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
        # self.vel = vec(0, 0)
        # keys = pg.key.get_pressed() # Keyboard events
        # if keys[pg.K_RIGHT]:
        #     self.vel.x += const.PLAYER_X_SPEED
        # elif keys[pg.K_LEFT]:
        #     self.vel.x += -const.PLAYER_X_SPEED
        # elif keys[pg.K_UP]:
        #     self.vel.y += -const.PLAYER_Y_SPEED
        # elif keys[pg.K_DOWN]:
        #     self.vel.y += const.PLAYER_Y_SPEED
            
        if np.array_equal(action, [1, 0, 0, 0]):
            self.vel.x = const.PLAYER_X_SPEED
        elif np.array_equal(action, [0, 1, 0, 0]):
            self.vel.x = -const.PLAYER_X_SPEED
        elif np.array_equal(action, [0, 0, 1, 0]):
            self.vel.y = -const.PLAYER_Y_SPEED
        elif np.array_equal(action, [0, 0, 0, 1]):
            self.vel.y = const.PLAYER_Y_SPEED
        # Updating pos
        self.pos += self.vel
        self.rect.center = self.pos

if __name__ == "__main__":
    pass




