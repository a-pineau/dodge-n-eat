"""Implements the game loop and handles the user's events."""

import os
import random
import pygame as pg

from itertools import cycle
from agent import Agent
from obstacle import Obstacle
from helper import message, distance
import constants as const
vec = pg.math.Vector2
n_snap = 0

# Manually places the window
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (100, 50)

MAX_FRAME = 1500
ENEMY_SIZE = 25
REWARD_CLOSE = 1
REWARD_DODGE = 5
REWARD_EAT = 10
REWARD_COLLISION = -10


class GameAI:
    def __init__(self) -> None:
        pg.init()
        self.screen = pg.display.set_mode([const.WIDTH, const.HEIGHT])
        pg.display.set_caption(const.TITLE)
        self.clock = pg.time.Clock()
        self.running = True
        self.n_frames = 0
        self.score = 0
        self.enemies = [
            Block(const.WIDTH*0.25, const.HEIGHT*0.25, ENEMY_SIZE),
            Block(const.WIDTH*0.75, const.HEIGHT*0.25, ENEMY_SIZE),
            Block(const.WIDTH*0.25, const.HEIGHT*0.75, ENEMY_SIZE),
            Block(const.WIDTH*0.75, const.HEIGHT*0.75, ENEMY_SIZE),
        ]
        self.agent = Agent(self)
        self.food = Food(self)
        self.distance_food = distance(self.agent.pos, self.food.pos)
        self.distance_enemy, self.closest_enemy = self.agent.closest_enemy()
        print(self.distance_enemy, self.closest_enemy)

    def reset(self):
        self.n_frames = 0
        self.score = 0
        self.agent.place()
        self.food.place()

    def play_step(self, action):
        self.n_frames += 1
        # events handler
        self.events()
        # updating position
        self.agent.update(action)
        # getting reward
        reward, game_over = self.get_reward()
        # displaying game
        self.display()
        # returning corresponding values
        return reward, game_over, self.score

    def get_reward(self) -> int:
        game_over = False
        reward = 0
        self.agent.closest_enemy()
        # positive reward if the agent gets closer to food between 2 frames
        self.old_distance_food = self.distance_food
        self.old_distance_enemy = self.distance_enemy
        self.old_closest_enemy = self.closest_enemy
        self.distance_enemy, self.closest_enemy = self.agent.closest_enemy()
        # if self.old_closest_enemy == self.closest_enemy:
        #     if self.old_distance_enemy > self.distance_enemy:
        #         print("HEAH")
        self.distance_food = distance(self.agent.pos, self.food.pos)
        if self.distance_enemy > self.old_distance_enemy:
            pass
        if self.distance_food < self.old_distance_food:
            reward = REWARD_CLOSE
        else:
            reward = -REWARD_CLOSE
            
        if self.n_frames > MAX_FRAME:
            game_over = True
        # checking for failure (wall or enemy collision)
        if self.agent.wall_collision(offset=0):
            reward = REWARD_COLLISION
            game_over = True
        if self.agent.enemy_collision(offset=0):
            reward = -20
            game_over = True
        # checking if eat:
        if self.agent.food_collision():
            self.n_frames = 0
            reward = REWARD_EAT
            self.score += 1
            self.food.place()
        return reward, game_over

    def events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
            if event.type == pg.KEYDOWN and event.key == pg.K_q:
                self.running = False

    def display(self):
        self.screen.fill(const.BACKGROUND)
        pg.draw.rect(self.screen, self.agent.color, self.agent.rect)
        pg.draw.rect(self.screen, self.food.color, self.food.rect)
        for enemy in self.enemies:
            pg.draw.rect(self.screen, enemy.color, enemy.rect)
        score_msg = f"Score: {self.score}"
        games_msg = f"Game: {self.agent.n_games}"
        message(self.screen, score_msg, 20, pg.Color("White"), (5, 5))
        message(self.screen, games_msg, 20, pg.Color("White"), (5, 30))
        pg.display.flip()
        self.clock.tick(const.FPS)


class Food(pg.sprite.Sprite):
    def __init__(self, game):
        pg.sprite.Sprite.__init__(self)
        self.game = game
        self.size = 15
        self.color = pg.Color("Green")
        self.pos = vec(const.WIDTH*0.85, const.HEIGHT*0.5)
        self.rect = pg.Rect(self.pos.x, self.pos.x, self.size*2, self.size*2)
        self.rect.center = self.pos
        self.place()

    def place(self):
        x = random.randint(self.size, const.WIDTH - self.size)
        y = random.randint(self.size, const.HEIGHT - self.size)
        self.pos = vec(x, y)
        self.rect = pg.Rect(self.pos.x, self.pos.x, self.size*2, self.size*2)
        self.rect.center = self.pos
        obstacles = [enemy.rect for enemy in self.game.enemies] + [self.game.agent.rect]
        collision_list = self.rect.collidelist(obstacles)
        # -1 is the return default value given by Pygame for the collidelist method if no collision found
        if collision_list != -1:
            print(f"Food collides with {collision_list}")
            self.place()


class Block(pg.sprite.Sprite):
    def __init__(self, x, y, size) -> None:
        pg.sprite.Sprite.__init__(self)
        self.size = size
        self.color = pg.Color("Red")
        self.pos = vec(x, y)
        self.rect = pg.Rect(self.pos.x, self.pos.x, self.size*2, self.size*2)
        self.rect.center = self.pos
