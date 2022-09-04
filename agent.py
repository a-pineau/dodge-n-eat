import torch
import random
import pygame as pg
import numpy as np
import constants as const

from collections import deque
from block import Block
from model import Linear_QNet, QTrainer

vec = pg.math.Vector2

MAX_MEMORY = 100_000  # Storing 100k max items in deque
BATCH_SIZE = 1000
N_INPUTS = 10
N_HIDDEN = 256
N_OUTPUTS = 4
LEARNING_RATE = 0.001
DECAY = True

EPSILON = 0.2
MAX_EPSILON = EPSILON
MIN_EPSILON = 0.001
DECAY = 0.01
DISCOUNT_FACTOR = 0.9

MOVES = {0: "right", 1: "left", 2: "down", 3: "up"}


class Agent(Block):
    def __init__(self, x, y, w, h, color, game, epsilon_decay=DECAY):
        super().__init__(x, h, w, h, color)
        self.game = game
        self.epsilon_decay = epsilon_decay
        self.direction = None
        self.decision = None
        self.last_decision = None
        self.color = pg.Color("Blue")
        self.place()

        # DQN
        self.epsilon = EPSILON
        self.max_epsilon = self.epsilon
        self.decay = DECAY
        self.n_exploration = 0
        self.n_exploitation = 0
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(
            input_size=N_INPUTS, hidden_size=N_HIDDEN, output_size=N_OUTPUTS
        )
        self.trainer = QTrainer(self.model, lr=LEARNING_RATE, gamma=DISCOUNT_FACTOR)

    def place(self):
        self.dangerous_locations = set()

        x = (const.PLAY_WIDTH + const.INFO_WIDTH) // 4
        y = const.PLAY_HEIGHT // 2

        self.pos = vec(x, y)
        self.rect.center = self.pos

    def wall_collision(self, offset):
        return (
            self.rect.left - offset < const.INFO_WIDTH
            or self.rect.right + offset > const.PLAY_WIDTH
            or self.rect.top - offset < 0
            or self.rect.bottom + offset > const.PLAY_HEIGHT
        )

    def enemy_collision(self):
        # If the agent is already colliding with an enemy, returns True directly
        if pg.sprite.spritecollide(self, self.game.enemies, False):
            return True

    def enemy_danger(self):
        offsets = [
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (1, 0),
            (-1, 1),
            (0, -1),
            (1, 1),
        ]

        for enemy in self.game.enemies:
            for offset in offsets:
                buffer_rect = self.rect.copy().move(offset)
                if buffer_rect.colliderect(enemy.rect):
                    return True

        return False

    def food_collision(self):
        return self.rect.colliderect(self.game.food.rect)

    def get_action(self, state):
        final_move = [0] * N_OUTPUTS
        random_number = random.random()

        if random_number <= self.epsilon:
            self.decision = "Exploration"
            self.n_exploration += 1
            action = random.randint(0, N_OUTPUTS - 1)
        else:
            self.decision = "Exploitation"
            self.n_exploitation += 1
            state_0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_0)
            action = torch.argmax(prediction).item()  # returns index of max value

        final_move[action] = 1
        return final_move

    def decay_epsilon(self):
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(
            -DECAY * self.game.n_games
        )

    def remember(self, state, action, reward, next_state, done):
        # will pop left if MAX_MEMORY is reached
        self.memory.append((state, action, reward, next_state, done))

    def replay_short(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def replay_long(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def update(self, action):
        # Tests only
        if self.game.human:
            keys = pg.key.get_pressed()  # Keyboard events
            if keys[pg.K_RIGHT]:
                self.direction = "right"
                self.pos.x += const.AGENT_X_SPEED
            elif keys[pg.K_LEFT]:
                self.direction = "left"
                self.pos.x += -const.AGENT_X_SPEED
            elif keys[pg.K_UP]:
                self.direction = "up"
                self.pos.y += -const.AGENT_Y_SPEED
            elif keys[pg.K_DOWN]:
                self.direction = "down"
                self.pos.y += const.AGENT_Y_SPEED
        else:
            if np.array_equal(action, [1, 0, 0, 0]):  # going right
                self.direction = "right"
                self.pos.x += const.AGENT_X_SPEED
            elif np.array_equal(action, [0, 1, 0, 0]):  # going left
                self.direction = "left"
                self.pos.x += -const.AGENT_X_SPEED
            elif np.array_equal(action, [0, 0, 1, 0]):  # going down
                self.direction = "up"
                self.pos.y += -const.AGENT_Y_SPEED
            elif np.array_equal(action, [0, 0, 0, 1]):  # going up
                self.direction = "down"
                self.pos.y += const.AGENT_Y_SPEED

        # Updating pos
        self.rect.center = self.pos


if __name__ == "__main__":
    pass
