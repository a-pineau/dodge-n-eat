final_move = [0, 0, 0, 0, 0]
rand_number = np.random.rand()

if rand_number <= self.epsilon:
    move = random.randint(0, N_OUTPUTS - 1)
else:
    state_0 = torch.tensor(state, dtype=torch.float)
    prediction = self.model(state_0)
    move = torch.argmax(prediction).item()  # returns index of max value

final_move[move] = 1
self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(
    -self.decay * self.n_games
)

return final_move
