import gym

from epsilon import Epsilon
from qtable import QTable
from qtable import train_qtable


def main():
    # step 1: loading the environment
    env = gym.make("FrozenLake-v0")

    # step 2: creating the Q-table
    state_size = env.observation_space.n
    action_size = env.action_space.n
    q = QTable(state_size, action_size)

    # step 3: creating de epsilon decay
    e = Epsilon(initial_epsilon=1.0, max_epsilon=1.0, min_epsilon=0.01, decay_rate=0.005)

    # step 4: Q-table training
    total_episodes = 100000
    max_steps = 100
    q, rewards = train_qtable(env, q, e, total_episodes, max_steps)

    print("Score over time {:.4f}".format(sum(rewards) / total_episodes))
    q.print()

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
