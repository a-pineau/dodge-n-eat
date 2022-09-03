from game import GameAI
from helper import plot


def train():
    game = GameAI(human=False, grid=False)
    agent = game.agent
    
    while game.running:
        # get old state
        prev_state = game.get_state()
        # get move (exploration or exploitation)
        action = agent.perform(prev_state)
        # play game and get new state
        reward, done = game.play_step(action)
        game.reward_episode += reward
        next_state = game.get_state()
        # remember
        agent.remember(prev_state, action, reward, next_state, done)
        # replaying previous games
        agent.replay()

        if done:
            if game.model.decay:
                game.model.decay_epsilon()
            game.n_games += 1
            game.reward_episode = 0

            game.sum_score += game.score
            game.sum_reward += game.reward_episode

            game.mean_rewards.append(game.sum_reward / game.n_games)
            game.mean_scores.append(game.sum_score / game.n_games)
            
            game.reset()

        # displaying game
        game.draw()

    # plotting
    plot(game.mean_scores, game.mean_rewards, "results.png")


if __name__ == "__main__":
    train()
