import numpy as np
import random


class QTable:
    """This class encapsulates a QTable, as well its update function."""

    def __init__(self, state_size, action_size, learning_rate=0.8, gamma=0.95):
        self._qtable = np.zeros((state_size, action_size))

        self._learning_rate = learning_rate
        self._gamma = gamma

    def select_action(self, env, state, epsilon=0.0):
        r = random.uniform(0, 1)

        if r < epsilon:
            return env.action_space.sample()

        return np.argmax(self._qtable[state, :])

    def update(self, state, action, reward, new_state):
        self._qtable[state, action] += self._learning_rate * \
            (reward + self._gamma * np.max(self._qtable[new_state, :]) - self._qtable[state, action])

    def print(self):
        np.set_printoptions(precision=4, suppress=True)
        print(self._qtable)


def train_qtable(env, qtable, epsilon, total_episodes, max_steps, verbose=True):
    rewards = []
    for episode in range(total_episodes):
        state = env.reset()
        total_rewards = 0

        for step in range(max_steps):
            action = qtable.select_action(env, state, epsilon.value)

            new_state, reward, done, info = env.step(action)

            qtable.update(state, action, reward, new_state)

            total_rewards += reward
            state = new_state

            if done:
                break

        if verbose and (episode % 1000 == 0):
            print("episode={0}, epsilon={1:.2f} score: {2}".format(episode, epsilon.value, total_rewards))

        epsilon.decay(episode)

        rewards.append(total_rewards)

    return qtable, rewards
