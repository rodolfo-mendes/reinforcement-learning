# Step 1: importing libraries

import tensorflow as tf              # Deep Learning Library
import numpy as np                   # Linear Algebra
import retro                         # Retro environment

from skimage import transform        # help us to preprocess our frames
from skimage.color import rgb2gray   # help us to gray our frames

import matplotlib.pyplot as plt      # display graphs

from collections import deque        # deque data structure

import random                        # random numbers

import warnings

warnings.filterwarnings('ignore')


# Step 2: creating the environment
env = retro.make(game='SpaceInvaders-Atari2600')

print("The size of our frame is: ", env.observation_space)
print("The action size is: ", env.action_space.n)

# encoded version of actions
possible_actions = np.array(np.identity(env.action_space.n, dtype=int).tolist())
print(possible_actions)


# Step 3: defining pre-processing functions
def preprocess_frame(frame):
    gray = rgb2gray(frame)

    cropped_frame = gray[8:-12, 4:-12]

    normalized_frame = cropped_frame / 255.0

    preprocessed_frame = transform.resize(normalized_frame, [110, 84])

    return preprocessed_frame


stack_size = 4
stacked_frames = deque([np.zeros((110, 84), dtype=int) for i in range(stack_size)], maxlen=4)


def stack_frames(stacked_frames, state, is_new_episode):
    frame = preprocess_frame(state)

    if is_new_episode:
        # clear out stacked frames
        stacked_frames = deque([np.zeros((110, 84), dtype=int) for i in range(stack_size)], maxlen=4)

        # Because we're in a new episode, stack the first frame 4 times
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
    else:
        stacked_frames.append(frame)

    # Stack the frames
    stacked_state = np.stack(stacked_frames, axis=2)

    return stacked_state, stacked_frames


# Step 4: setup hyperparameters

# MODEL HYPERPARAMETERS
state_size = [110, 84, 4]         # our input is a stack of 4 frames, hence 110x84x4 (width, height, channels)
action_size = env.action_space.n  # 8 possible actions
learning_rate = 0.00025           # Alpha (learning rate)

# TRAINING HYPERPARAMETERS
total_episodes = 1               # total episodes for training
max_steps = 100                 # max steps for a episode

# total_episodes = 10               # total episodes for training
# max_steps = 100                 # max steps for a episode


batch_size = 64                   # batch size

# Exploration parameters for epsilon-greedy
explore_start = 1.0               # exploration probability at the start
explore_stop = 0.1                # minimum exploration probability
decay_rate = 0.00001              # exponential decay rate for exploration prob

# Q Learning hyperparameters
gamma = 0.9                       # discounting rate

# MEMMORY HYPERPARAMETERS
pretrain_length = batch_size      # number of experiences stored in memory when initialized
memory_size = 1000000             # maximum number of experiences memory can keep

# PREPROCESSING HYPERPARAMETERS
stack_size = 4                    # number of frames stacked

# MODIFY THIS TO False IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = True

# TURN THIS TO True IF YOU WANT TO RENDER THE ENVIRONMENT
episode_render = False


# Step 5: Create our Deep Q-Learning Neural Network model
class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            self.inputs = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions = tf.placeholder(tf.float32, [None, self.action_size], name="actions")

            self.target_q = tf.placeholder(tf.float32, [None], name="target")

            """
            First convolutional net:
            CNN
            ELU
            """
            # Input is 110x84x4
            self.conv1 = tf.layers.conv2d(inputs=self.inputs,
                                          filters=32,
                                          kernel_size=[8, 8],
                                          strides=[4, 4],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv1")
            self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")

            """
            Second convolutional net:
            CNN
            ELU
            """
            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out,
                                          filters=64,
                                          kernel_size=[4, 4],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv2")
            self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")

            """
            Third convolutional net:
            CNN
            ELU
            """
            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out,
                                          filters=64,
                                          kernel_size=[3, 3],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv3")
            self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")

            self.flatten = tf.contrib.layers.flatten(self.conv3_out)

            self.fc = tf.layers.dense(inputs=self.flatten,
                                      units=512,
                                      activation=tf.nn.elu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      name="fc1")

            self.output = tf.layers.dense(inputs=self.fc,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units=self.action_size,
                                          activation=None)

            self.q = tf.reduce_sum(tf.multiply(self.output, self.actions))

            self.loss = tf.reduce_mean(tf.square(self.target_q - self.q))

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


# Reset the graph
tf.reset_default_graph()

# Instantiate the DQNetwork
dq_network = DQNetwork(state_size, action_size, learning_rate)


# Step 6: experience replay
class Memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size), size=batch_size, replace=False)
        return [self.buffer[i] for i in index]


# Instantiate memory
memory = Memory(max_size=memory_size)
for i in range(pretrain_length):
    if i == 0:
        state = env.reset()
        state, stacked_frames = stack_frames(stacked_frames, state, True)

    choice = random.randint(1, len(possible_actions)) - 1
    action = possible_actions[choice]
    next_state, reward, done, _ = env.step(action)

    # Stack the frames
    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

    # If the episode is finished (dead 3x)
    if done:
        # we finished the episode
        next_state = np.zeros(state.shape)

        # add experience to memory
        memory.add((state, action, reward, next_state, done))

        # start a new episode
        state = env.reset()

        # stack the frames
        state, stacked_frames = stack_frames(stacked_frames, state, True)
    else:
        # add experience to memory
        memory.add((state, action, reward, next_state, done))

        # our new state is now the next_state
        state = next_state

# Step 7: setup TensorBoard

writer = tf.summary.FileWriter("./tensorboard/dqn/1")

# Losses
tf.summary.scalar("Loss", dq_network.loss)

write_op = tf.summary.merge_all()


# Step 8: training the agent
def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions, sess):
    exp_exp_tradeoff = np.random.rand()

    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

    if (explore_probability > exp_exp_tradeoff):
        # make a random action (exploration)
        choice = random.randint(1, len(possible_actions) - 1)
        action = possible_actions[choice]
    else:
        qs = sess.run(dq_network.output, feed_dict={dq_network.inputs: state.reshape((1, *state.shape))})

        # take the biggest q value
        choice = np.argmax(qs)
        action = possible_actions[choice]

    return action, explore_probability


# Saver will help us save our model
saver = tf.train.Saver()

if training:
    with tf.Session() as sess:
        # initialize the variables
        sess.run(tf.global_variables_initializer())

        # initialize the decay rate (that will use to reduce epsilon)
        decay_step = 0

        # list of rewards by episode
        rewards_list = []

        for episode in range(total_episodes):
            # set step to 0
            step = 0

            # initialize the rewards of the episode
            episode_rewards = []

            # initialize loss
            loss = 0

            # make a new episode and observe  the first state
            state = env.reset()

            # remember that stack frame function also call our preprocess function
            state, stacked_frames = stack_frames(stacked_frames, state, True)

            while step < max_steps:
                step += 1

                # increase decay_step
                decay_step += 1

                # predict the action to be taked and take it
                action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state,
                                                             possible_actions, sess)

                # perform action and get the next state
                next_state, reward, done, _ = env.step(action)

                if episode_render:
                    env.render()

                # if the game is finished
                if done:
                    # the episode ends so no next state
                    next_state = np.zeros((110, 84), dtype=np.int)

                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                    # Set step = max_steps to end episode
                    step = max_steps

                    # Get the total reward of the episode
                    total_reward = np.sum(episode_rewards)

                    print("Episode: {} Total reward: {} Explore P: {:.4f} Training Loss: {:.4f}"
                          .format(episode, total_reward, explore_probability, loss))

                    rewards_list.append((episode, total_reward))

                    # store transition <st, at, rt+1, st+1> in memory
                    memory.add((state, action, reward, next_state, done))
                else:
                    # stack the frame of the next state
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                    # add experience to the memory
                    memory.add((state, action, reward, next_state, done))

                    # st+1 is now our current state
                    state = next_state

                # LEARNING PART
                # obtain random mini-batch from memory
                batch = memory.sample(batch_size)
                states_mb = np.array([each[0] for each in batch], ndmin=3)
                actions_mb = np.array([each[1] for each in batch])
                rewards_mb = np.array([each[2] for each in batch])
                next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                dones_mb = np.array([each[4] for each in batch])

                target_qs_batch = []

                # Get Q values for next state
                qs_next_state = sess.run(dq_network.output, feed_dict={dq_network.inputs: next_states_mb})

                # set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma + maxQ(s', a')
                for i in range(len(batch)):
                    terminal = dones_mb[i]

                    # if we are in a terminal state, only equals reward
                    if terminal:
                        target_qs_batch.append(rewards_mb[i])
                    else:
                        target = rewards_mb[i] + gamma * np.max(qs_next_state[i])
                        target_qs_batch.append(target)

                targets_mb = np.array([each for each in target_qs_batch])

                loss, _ = sess.run([dq_network.loss, dq_network.optimizer],
                                   feed_dict={dq_network.inputs: states_mb,
                                              dq_network.target_q: targets_mb,
                                              dq_network.actions: actions_mb})

                # Write TF Summaries
                summary = sess.run(write_op, feed_dict={dq_network.inputs: states_mb,
                                                        dq_network.target_q: targets_mb,
                                                        dq_network.actions: actions_mb})

                writer.add_summary(summary, episode)
                writer.flush()

            # save model at every 5 episodes
            if episode % 5 == 0:
                save_path = saver.save(sess, "./models/model.ckpt")
                print("Model saved")


# Step 9: test and watch our agent play
with tf.Session() as sess:
    total_test_rewards = []

    # Load the model
    saver.restore(sess, "./models/model.ckpt")

    for episodes in range(1):
        total_rewards = 0

        state = env.reset()
        state, stacked_frames = stack_frames(stacked_frames, state, True)

        print("******************************")
        print(" EPISODE ", episode)

        while True:
            state = state.reshape((1, *state_size))

            qs = sess.run(dq_network.output, feed_dict={dq_network.inputs: state})

            choice = np.argmax(qs)
            action = possible_actions[choice]

            next_state, reward, done, _ = env.step(action)
            env.render()

            total_rewards + reward

            if done:
                print("Score", total_rewards)
                total_test_rewards.append(total_reward)

            next_state, stacked_frames = stacked_frames(stacked_frames, next_state, False)
            state = next_state

env.close()
