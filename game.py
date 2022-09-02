"""Implements the game loop and handles the user's events."""

import os
import random
from tarfile import BLOCKSIZE
import numpy as np
import pygame as pg

from itertools import cycle
from agent import Agent
from obstacle import Obstacle
from helper import message, distance
import constants as const
vec = pg.math.Vector2
n_snap = 0

# Manually places the window
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (50, 50)

MAX_FRAME = 250

REWARD_CLOSE_WALL = 5
REWARD_CLOSE_FOOD = 1
REWARD_EAT = 10

PENALTY_WANDER = -1
PENALTY_COLLISION = -10
PENALTY_FAR_FOOD = -0.1


class GameAI:
    def __init__(self, human=False, grid=False) -> None:
        pg.init()
        self.human = human
        self.grid = grid
        self.screen = pg.display.set_mode([const.WIDTH, const.HEIGHT])
        self.clock = pg.time.Clock()

        pg.display.set_caption(const.TITLE)

        self.running = True
        self.n_frames = 0
        self.n_frames_threshold = 0
        self.score = 0
        self.highest_score = 0
        self.reward_episode = 0

        self.enemies = [
            Enemy(const.WIDTH // 2, const.HEIGHT // 2, 
                  const.BLOCK_SIZE * 7, const.BLOCK_SIZE * 7)
        ]

        self.agent = Agent(self)
        self.food = Food(self)
        self.distance_food = distance(self.agent.pos, self.food.pos)
    
    @staticmethod
    def set_global_seed(seed: int) -> None:
        """
        Sets random seed into PyTorch, numpy and random.

        Args:
            seed: random seed
        """

        try:
            import torch
        except ImportError:
            print("Module PyTorch cannot be imported")
            pass
        else:
            torch.manual_seed(seed)
            if torch.cuda.is_available(): 
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        random.seed(seed)
        np.random.seed(seed) 
        
    def reset(self):
        self.n_frames_threshold = 0
        self.score = 0
        self.reward_episode = 0
        self.agent.place()
        self.food.place()

    def play_step(self, action):
        self.n_frames_threshold += 1

        self.events()
        self.agent.update(action)

        # returning corresponding values
        reward, game_over = self.get_reward()
        self.reward_episode += reward
        return reward, game_over, self.score

    def get_reward(self) -> tuple:
        game_over = False
        reward = 0

        # stops episode if the agent does nothing but wonder around
        if self.n_frames_threshold > MAX_FRAME:
            return PENALTY_WANDER, True

        # checking for failure (wall or enemy collision)
        if self.agent.wall_collision(offset=0) or self.agent.enemy_collision():
            return PENALTY_COLLISION, True

        # checking if agent is getting closer to food
        self.old_distance_food = self.distance_food
        self.distance_food = distance(self.agent.pos, self.food.pos)
        if self.distance_food < self.old_distance_food:
            reward = REWARD_CLOSE_FOOD
        else:
            reward = PENALTY_FAR_FOOD

        # checking for any enemy nearby and tag its location as dangerous
        if self.agent.enemy_danger():
            if self.agent.rect.center not in self.agent.dangerous_locations:
                self.agent.dangerous_locations.add(self.agent.rect.center)
                reward = 1
            else:
                reward = -1

        # checking if eat:
        if self.agent.food_collision():
            reward = REWARD_EAT
            self.score += 1
            self.n_frames_threshold = 0
            self.food.place()

        return reward, game_over

    def events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
            if event.type == pg.KEYDOWN and event.key == pg.K_q:
                self.running = False

    def display(self, mean_scores):
        self.screen.fill(const.BACKGROUND_COLOR)

        # Drawing blocks
        for enemy in self.enemies:
            enemy.draw(self.screen)

        pg.draw.rect(self.screen, self.agent.color, self.agent.rect)
        pg.draw.rect(self.screen, self.food.color, self.food.rect)

        # Drawing grid
        if self.grid:
            for i in range(1, const.WIDTH // const.BLOCK_SIZE):
                # horizontal lines
                start_h = const.BLOCK_SIZE * i, 0
                end_h = const.BLOCK_SIZE * i, const.HEIGHT
                
                # vertical lines
                start_v = 0, const.BLOCK_SIZE * i
                end_v = const.WIDTH, const.BLOCK_SIZE * i

                pg.draw.line(self.screen, const.GRID_COLOR, start_h, end_h)
                pg.draw.line(self.screen, const.GRID_COLOR, start_v, end_v)

        # Infos texts
        if self.score > self.highest_score:
            self.highest_score = self.score

        try:
            mean_score = round(mean_scores[-1], 1)
        except IndexError:
            mean_score = 0.0

        perc_exploration = (
            self.agent.n_exploration
            / (self.agent.n_exploration + self.agent.n_exploitation)
            * 100)
        perc_exploitation = (
            self.agent.n_exploitation
            / (self.agent.n_exploration + self.agent.n_exploitation)
            * 100)
        perc_threshold = int((self.n_frames_threshold / MAX_FRAME) * 100)

        infos = [
            f"Game: {self.agent.n_games}",
            f"Reward game: {round(self.reward_episode, 1)}",
            f"Score: {self.score}",
            f"Highest score: {self.highest_score}",
            f"Mean score: {mean_score}",
            f"Epsilon: {round(self.agent.epsilon, 4)}",
            f"Exploration: {round(perc_exploration, 3)}%",
            f"Exploitation: {round(perc_exploitation, 3)}%",
            f"Last decision: {self.agent.last_decision}",
            f"Threshold: {perc_threshold}%",
            f"Time: {int(pg.time.get_ticks() / 1e3)}s",
            f"FPS: {int(self.clock.get_fps())}",
        ]

        # Drawing infos
        for i, info in enumerate(infos):
            message(
                self.screen,
                info,
                const.INFOS_SIZE,
                const.INFOS_COLOR,
                (5, 5 + i * const.Y_OFFSET_INFOS)
            )

        pg.display.flip()
        self.clock.tick(const.FPS)


class Food(pg.sprite.Sprite):
    def __init__(self, game):
        pg.sprite.Sprite.__init__(self)
        self.game = game
        self.size = const.BLOCK_SIZE
        self.color = pg.Color("Green")
        self.place()

    def place(self):
        idx_x = random.randint(1, (const.WIDTH // const.BLOCK_SIZE) - 1)
        idx_y = random.randint(1, (const.HEIGHT // const.BLOCK_SIZE) - 1)
        x = idx_x * const.BLOCK_SIZE
        y = idx_y * const.BLOCK_SIZE

        self.pos = vec(x, y)
        self.rect = pg.Rect(self.pos.x, self.pos.y, self.size, self.size)

        # Checking for potential collisions with other elements
        obstacles = [enemy.rect for enemy in self.game.enemies] + \
            [self.game.agent.rect]
        collision_list = self.rect.collidelist(obstacles)

        # -1 is the return default value given by Pygame for the collidelist method if no collision found
        if collision_list != -1:
            self.place()


class Enemy(pg.sprite.Sprite):
    def __init__(self, x, y, w, h, color=pg.Color("Red")):
        pg.sprite.Sprite.__init__(self)
        
        self.pos = vec(x, y)
        self.color = color
        self.image = pg.Surface((w, h))
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.center = self.pos
        
    def draw(self, screen):
        pg.draw.rect(screen, self.color, self.rect)
        
