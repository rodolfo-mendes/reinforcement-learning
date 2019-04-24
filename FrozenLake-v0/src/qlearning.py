# step 0: importing dependencies

import numpy as np
import gym
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


class Epsilon:
    def __init__(self, initial_epsilon=1.0, max_epsilon=1.0, min_epsilon=0.01, decay_rate=0.005):
        self._epsilon = initial_epsilon
        self._max = max_epsilon
        self._min = min_epsilon
        self._decay_rate = decay_rate

    def decay(self, i):
        self._epsilon = self._min + (self._max - self._min) * np.exp(-self._decay_rate * i)
        return self._epsilon

    @property
    def value(self):
        return self._epsilon


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


def main():
    # step 1: loading the environment
    env = gym.make("FrozenLake-v0")

    # hyperparameters
    total_episodes = 100000
    max_steps = 100

    # step 2: creating the Q-table
    state_size = env.observation_space.n
    action_size = env.action_space.n

    q = QTable(state_size, action_size)

    q, rewards = train_qtable(env, q, Epsilon(), total_episodes, max_steps)

    print("Score over time {:.4f}".format(sum(rewards) / total_episodes))
    np.set_printoptions(precision=4, suppress=True)
    print(q._qtable)

    # Play
    env.reset()

    rewards = []

    for episode in range(1000):
        state = env.reset()
        step = 0
        total_rewards = 0

        for step in range(100):
            action = q.select_action(env, state)

            new_state, reward, done, info = env.step(action)

            total_rewards += reward
            state = new_state

            if done:
                break

        rewards.append(total_rewards)

        if episode % 100 == 0:
            print("******************************************")
            print("EPISODE {}".format(episode))
            print("Number of steps: {}".format(step))
            env.render()

    print("Score over time {:.4f}".format(sum(rewards) / 1000))

    env.close()


if __name__ == "__main__":
    main()
