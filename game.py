"""Implements the game loop and handles the user's events."""

import os
import random
from tarfile import BLOCKSIZE
import numpy as np
import pygame as pg

from agent import Agent
from block import Block
from helper import message, distance
import constants as const

vec = pg.math.Vector2
n_snap = 0

# Manually places the window
os.environ["SDL_VIDEO_WINDOW_POS"] = "%d,%d" % (50, 50)

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
        self.screen = pg.display.set_mode([const.PLAY_WIDTH, const.PLAY_HEIGHT])
        self.clock = pg.time.Clock()

        pg.display.set_caption(const.TITLE)

        self.running = True
        self.n_games = 0
        self.n_frames_threshold = 0
        self.score = 0
        self.highest_score = 0
        self.sum_scores = 0
        self.sum_rewards = 0
        self.mean_scores = [0]
        self.mean_rewards = [0]
        self.reward_episode = 0

        self.enemies = [
            # Block(
            #     (const.INFO_WIDTH + const.PLAY_WIDTH) / 3,
            #     const.PLAY_HEIGHT // 2,
            #     const.BLOCK_SIZE * 1,
            #     const.BLOCK_SIZE * 11,
            # ),
        ]
        self.agent = Agent(
            const.X_AGENT,
            const.Y_AGENT,
            const.BLOCK_SIZE,
            const.BLOCK_SIZE,
            pg.Color("Blue"),
            self,
        )
        self.food = Block(0, 0, const.BLOCK_SIZE, const.BLOCK_SIZE, pg.Color("Green"))
        self.place_food()
        self.distance_food = distance(self.agent.pos, self.food.pos)

    ####### Methods #######

    def place_food(self):
        idx_x = random.randint(
            1, ((const.PLAY_WIDTH - const.INFO_WIDTH) // const.BLOCK_SIZE) - 1
        )
        idx_y = random.randint(1, (const.PLAY_HEIGHT // const.BLOCK_SIZE) - 1)
        x = const.INFO_WIDTH + idx_x * const.BLOCK_SIZE
        y = idx_y * const.BLOCK_SIZE

        self.food.pos = vec(x, y)
        self.food.rect.center = self.food.pos

        # Checking for potential collisions with other entities
        obstacles = [enemy.rect for enemy in self.enemies] + [self.agent.rect]
        if self.food.rect.collidelist(obstacles) != -1:
            self.place_food()

    def reset(self):
        self.n_frames_threshold = 0
        self.score = 0
        self.reward_episode = 0
        self.agent.place()
        self.place_food()

    def play_step(self, action):
        self.n_frames_threshold += 1

        self.events()
        self.agent.update(action)

        # returning corresponding values
        reward, game_over = self.get_reward()
        self.reward_episode += reward
        return reward, game_over, self.score

    def get_state(self) -> np.array:
        return np.array(
            [
                # current direction
                self.agent.direction == "right",
                self.agent.direction == "left",
                self.agent.direction == "down",
                self.agent.direction == "up",
                # food location
                self.agent.rect.right <= self.food.rect.left,  # food is right
                self.agent.rect.left >= self.food.rect.right,  # food is left
                self.agent.rect.bottom <= self.food.rect.top,  # food is bottom
                self.agent.rect.top >= self.food.rect.bottom,  # food is up
                # dangers
                self.agent.enemy_danger(),
                self.agent.wall_collision(offset=const.BLOCK_SIZE),
            ],
            dtype=int,
        )

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
            self.place_food()

        return reward, game_over

    def events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
            if event.type == pg.KEYDOWN and event.key == pg.K_q:
                self.running = False

    def draw(self):
        """TODO"""
        self.screen.fill(const.BACKGROUND_COLOR)
        self.draw_entities()
        if self.grid:
            self.draw_grid()
        self.draw_infos()

        pg.display.flip()
        self.clock.tick(const.FPS)

    def draw_entities(self):
        """TODO"""
        for enemy in self.enemies:
            enemy.draw(self.screen)

        pg.draw.rect(self.screen, self.agent.color, self.agent.rect)
        pg.draw.rect(self.screen, self.food.color, self.food.rect)

    def draw_grid(self):
        """TODO"""
        for i in range(1, const.PLAY_WIDTH // const.BLOCK_SIZE):
            # vertical lines
            p_v1 = const.INFO_WIDTH + const.BLOCK_SIZE * i, 0
            p_v2 = const.INFO_WIDTH + const.BLOCK_SIZE * i, const.PLAY_HEIGHT

            # horizontal lines
            p_h1 = 0, const.BLOCK_SIZE * i
            p_h2 = const.PLAY_WIDTH, const.BLOCK_SIZE * i

            pg.draw.line(self.screen, const.GRID_COLOR, p_v1, p_v2)
            pg.draw.line(self.screen, const.GRID_COLOR, p_h1, p_h2)

    def draw_infos(self):
        """Draws game informations"""

        if self.score > self.highest_score:
            self.highest_score = self.score

        perc_exploration = (
            self.agent.n_exploration
            / (self.agent.n_exploration + self.agent.n_exploitation)
            * 100
        )
        perc_exploitation = (
            self.agent.n_exploitation
            / (self.agent.n_exploration + self.agent.n_exploitation)
            * 100
        )
        perc_threshold = int((self.n_frames_threshold / MAX_FRAME) * 100)

        infos = [
            f"Game: {self.n_games}",
            f"Reward game: {round(self.reward_episode, 1)}",
            f"Mean reward: {round(self.mean_rewards[-1], 1)}",
            f"Score: {self.score}",
            f"Highest score: {self.highest_score}",
            f"Mean score: {round(self.mean_scores[-1], 1)}",
            f"Initial Epsilon: {self.agent.max_epsilon}",
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
                (5, 5 + i * const.Y_OFFSET_INFOS),
            )

        # sep line
        pg.draw.line(
            self.screen,
            const.SEP_LINE_COLOR,
            (const.INFO_WIDTH, 0),
            (const.INFO_WIDTH, const.INFO_HEIGHT),
        )


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


def main():
    pass


if __name__ == "__main__":
    main()
