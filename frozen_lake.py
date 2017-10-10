#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
import sys
from collections import deque

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import clone_model, load_model
from keras.utils.np_utils import to_categorical

from model import create_q

parser = argparse.ArgumentParser()
parser.add_argument('--save', action='store_true')
parser.add_argument('--buffer_size', '-b', type=int, default=1_000_000)
parser.add_argument('--new', action='store_true')
parser.add_argument('--iterations', '-n', type=int, default=1_000_000)
parser.add_argument('--discount', '-d', type=float, default=.995)
parser.add_argument('--exploration_rate', '-e', type=float, default=.50)
args = parser.parse_args()

# TODO  Double DQN
# TODO  importance sample minibatches
# TODO  Dueling DQN
# TODO use Embedding layer

env = gym.make('FrozenLake-v0')

S = env.observation_space.n
A = env.action_space.n
TERMINAL_STATE = S - 1  # XXX Env specific.

# Use `deque` because it's efficient to remove the leading elements to expire them.

epsilon = args.exploration_rate
gamma = args.discount

ITERS = args.iterations
RUNNING_REWARDS_ITERS = 1_000

BATCH_SIZE = 1024
BUFFER_SIZE = args.buffer_size
TRAIN_SIZE = 20 * BATCH_SIZE  # train on samples of 10 minibatches

VERBOSE = False

# automatically handles expiring elements
buffer = deque(maxlen=BUFFER_SIZE)

Q = load_model('model.h5') if not args.new and os.path.exists('model.h5') else create_q(S, A)
target = load_model('target.h5') if not args.new and os.path.exists('target.h5') else clone_model(Q)


def eps_greedy(s: np.int64, epsilon=epsilon):
    # `np.argmax` works along flattened array, so for an nested array with a single entry, we get the right answer.
    return np.argmax(Q.predict(to_categorical(s, S))
                     ) if random.random() > epsilon else env.action_space.sample()


def get_batches(data):
    # One-hot encode states.
    states = to_categorical(data[:, 0], num_classes=S).astype(np.float32)
    succ_states = to_categorical(data[:, 3], num_classes=S).astype(np.float32)

    # Actions must be ints so we can use them as indices.
    actions = data[:, 1].astype(np.int)
    rewards = data[:, 2].astype(np.float32)

    td_estimates = target.predict(states)

    for td_estimate, a, r, s_ in zip(td_estimates, actions, rewards, succ_states):
        # `Q.predict` returns a (1,A) array, so we use [0] to extract the sub-array.
        td_estimates[a] += r + gamma * np.max(target.predict(s_[None, :]))
    return states, td_estimates


if __name__ == '__main__':

    running_rews = []  # XXX Env specific.

    for i in range(ITERS):

        # Decay exploration over time up to a baseline
        epsilon = max(.05, .99 * epsilon)

        done = False

        s = env.reset()

        while not done:
            a = eps_greedy(s)
            s_, r, done, _ = env.step(a)
            buffer.append([s, a, r, s_])
            s = s_

        # All terminal states have 0 reward and loop back to themselves as a successor state for all actions.
        # XXX Env specific.
        for a in range(A):
            buffer.append([TERMINAL_STATE, a, 0, TERMINAL_STATE])

        # XXX Env specific.
        # Get last reward as score for whole episode to calculate OpenAI score.
        running_rews.append(r)

        # XXX Env specific.
        if i % RUNNING_REWARDS_ITERS == 0:
            num_successes = sum(1 for r in running_rews if r > 0)
            print(f'Score: {np.mean(running_rews)}')
            running_rews.clear()

        # Takes about 5000 iterations to gather enough data.
        if i % 1_000 == 0 and len(buffer) >= TRAIN_SIZE:
            # Sample from buffer.
            data = np.array(random.sample(buffer, k=TRAIN_SIZE))

            states, td_estimates = get_batches(data)

            Q.fit(
                x=states,
                y=td_estimates,
                batch_size=BATCH_SIZE,
                validation_split=0.1,
                verbose=VERBOSE,
            )

        if i % 5_000 == 0 and i > 0:
            # copy Q to update target
            target = clone_model(Q)

        if args.save and i % 10_000 == 0 and i > 0:

            Q.save('model.h5')
            target.save('target.h5')
