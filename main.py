from game import GameAI
from helper import plot

MAX_N_GAMES = 60_000


def train():
    game = GameAI(human=False, grid=False)
    agent = game.agent

    while game.running:
        if game.n_games > MAX_N_GAMES:
            break

        # get old state
        state = game.get_state()
        # get move (exploration or exploitation)
        action = agent.get_action(state)
        # play game and get new state
        reward, done, score = game.play_step(action)
        new_state = game.get_state()
        game.sum_rewards += reward
        # train short memory
        agent.replay_short(state, action, reward, new_state, done)
        # remember
        agent.remember(state, action, reward, new_state, done)

        if done:
            if agent.epsilon_decay:
                agent.decay_epsilon()
                
            game.agent.last_decision = game.agent.decision
            game.n_games += 1
            game.agent.replay_long()
            game.reset()

            game.sum_scores += score
            game.mean_scores.append(game.sum_scores / game.n_games)
            game.mean_rewards.append(game.sum_rewards / game.n_games)

        # displaying game
        game.draw()

    # plotting
    # plot(game.mean_scores, game.mean_rewards, "results.png")


if __name__ == "__main__":
    train()
