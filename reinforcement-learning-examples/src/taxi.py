import gym

from epsilon import Epsilon
from qtable import QTable, train_qtable


def main():
    # Step 1: create the Taxi-v2 environment
    env = gym.make("Taxi-v2")

    # Step 2: create the QTable
    q = QTable(env.observation_space.n, env.action_space.n, learning_rate=0.7, gamma=0.99)

    # Step 3: create the Epsilon decay
    e = Epsilon()

    # Step 4: Q-table training
    total_episodes = 100000
    max_steps = 100
    q, rewards = train_qtable(env, q, e, total_episodes, max_steps, verbose=True)

    print("Score over time {:.4f}".format(sum(rewards) / total_episodes))
    q.print()

    env.render()

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


if __name__ == '__main__':
    main()
