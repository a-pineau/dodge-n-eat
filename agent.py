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
LR = 0.001
MOVES = ["right", "left"]
X_AGENT, Y_AGENT = 100, 200

class Agent(pg.sprite.Sprite):
    # -----------
    def __init__(self, game):
        pg.sprite.Sprite.__init__(self)
        self.game = game
        self.size = 15
        self.place()
        self.vel = vec(0, 0)
        self.color = pg.Color("Blue")
        # DQN
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(input_size=12, hidden_size=256, output_size=4)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma) 

    def place(self):
        self.pos = vec(X_AGENT, Y_AGENT)
        self.rect = pg.Rect(self.pos.x, self.pos.y, self.size*2, self.size*2)
        self.rect.center = self.pos

    def wall_collision(self, offset):
        # left/right collision
        return (
            self.rect.left - offset < 0 or 
            self.rect.right + offset > const.WIDTH or
            self.rect.top - offset < 0 or
            self.rect.bottom + offset > const.HEIGHT
        )

    def enemy_collision(self, offset):
        collision = False
        # If the agent is already colliding with an enemy, returns True directly
        if pg.sprite.spritecollide(self, self.game.enemies, False):
            return True
        # Otherwise, checking in all 4 directions for a danger
        self.rect.left -= offset
        if pg.sprite.spritecollide(self, self.game.enemies, False):
            collision = True
        self.rect.left += offset

        self.rect.right += offset
        if pg.sprite.spritecollide(self, self.game.enemies, False):
            collision = True
        self.rect.right -= offset

        self.rect.bottom += offset
        if pg.sprite.spritecollide(self, self.game.enemies, False):
            collision = True
        self.rect.bottom -= offset
        
        self.rect.top -= offset
        if pg.sprite.spritecollide(self, self.game.enemies, False):
            collision = True
        self.rect.top += offset

        return collision

    def food_collision(self):
        return self.rect.colliderect(self.game.food.rect)

    def get_state(self) -> np.array:
        # dangers
        close_collision = (
            self.wall_collision(offset=self.size) or 
            self.enemy_collision(offset=self.size)
        )
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
            going_right and close_collision,
            going_left and close_collision,
            going_up and close_collision,
            going_down and close_collision,

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
            move = random.randint(0, 3)
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
        self.vel = vec(0, 0)
        keys = pg.key.get_pressed() # Keyboard events
        if keys[pg.K_RIGHT]:
            self.vel.x += const.PLAYER_X_SPEED
        elif keys[pg.K_LEFT]:
            self.vel.x += -const.PLAYER_X_SPEED
        elif keys[pg.K_UP]:
            self.vel.y += -const.PLAYER_Y_SPEED
        elif keys[pg.K_DOWN]:
            self.vel.y += const.PLAYER_Y_SPEED
        # if np.array_equal(action, [1, 0, 0, 0]):
        #     self.vel.x = const.PLAYER_X_SPEED
        # elif np.array_equal(action, [0, 1, 0, 0]):
        #     self.vel.x = -const.PLAYER_X_SPEED
        # elif np.array_equal(action, [0, 0, 1, 0]):
        #     self.vel.y = -const.PLAYER_Y_SPEED
        # elif np.array_equal(action, [0, 0, 0, 1]):
        #     self.vel.y = const.PLAYER_Y_SPEED
        # Updating pos
        self.pos += self.vel
        self.rect.center = self.pos

if __name__ == "__main__":
    pass




