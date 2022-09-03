from game import GameAI
from helper import plot

MAX_N_GAMES = 60_000


def train():
    sum_scores = 0
    sum_rewards = 0
    highest_score = 0
    mean_rewards = []
    mean_scores = []
    game = GameAI(human=False, grid=False)

    while game.running:
        if game.agent.n_games > MAX_N_GAMES:
            break

        # get old state
        state = game.agent.get_state()
        # get move (exploration or exploitation)
        final_move = game.agent.get_action(state)
        # play game and get new state
        reward, done, score = game.play_step(final_move)
        new_state = game.agent.get_state()
        sum_rewards += reward
        # train short memory
        game.agent.train_short_memory(state, final_move, reward, new_state, done)
        # remember
        game.agent.remember(state, final_move, reward, new_state, done)

        if done:
            game.agent.last_decision = game.agent.decision
            game.agent.n_games += 1
            game.agent.train_long_memory()
            game.reset()

            sum_scores += score
            mean_scores.append(sum_scores / game.agent.n_games)
            mean_rewards.append(sum_rewards / game.agent.n_games)

        # displaying game
        game.display(mean_scores, mean_rewards)

    # plotting
    plot(mean_scores, mean_rewards, "results.png")


if __name__ == "__main__":
    train()
