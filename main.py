import pygame as pg
import numpy as np
from game import GameAI
from helper import plot

def train():
    cummulative_score = 0
    cummulative_reward = 0
    mean_rewards = []
    scores = []
    mean_scores = []
    game = GameAI()
    while game.running:
        if game.agent.n_games > 100:
            break
        # get old state
        old_state = game.agent.get_state()
        # get move (exploration or exploitation)
        final_move = game.agent.get_action(old_state)
        # play game and get new state
        reward, done, score = game.play_step(final_move)
        new_state = game.agent.get_state()
        if np.any(old_state[:4]) and not np.any(new_state[:4]):
            print("dodged")
        cummulative_reward += reward
        # train short memory
        game.agent.train_short_memory(old_state, final_move, reward, new_state, done)
        # remember
        game.agent.remember(old_state, final_move, reward, new_state, done)
        if done:
            print(f"game: {game.agent.n_games} | score = {score} | reward (sum) = {cummulative_reward}")
            cummulative_score += score
            game.reset()
            game.agent.n_games += 1
            game.agent.train_long_memory()
            scores.append(score)
            mean_scores.append(cummulative_score/game.agent.n_games)
            mean_rewards.append(cummulative_reward/game.agent.n_games)

    # plotting
    plot(scores, mean_scores, "results_vel_7_offset_size.png")


if __name__ == "__main__":
    train()
