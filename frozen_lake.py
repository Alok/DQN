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
from keras.models import load_model
from keras.utils.np_utils import to_categorical

from model import create_q

parser = argparse.ArgumentParser()
parser.add_argument('--save', action='store_true')
parser.add_argument('--buffer_size', '-b', type=int, default=25_000)
parser.add_argument('--new', action='store_true')
parser.add_argument('--iterations', '-n', type=int, default=1_000_000)
parser.add_argument('--discount', '-d', type=float, default=.999)
parser.add_argument('--exploration_rate', '-e', type=float, default=.90)
args = parser.parse_args()

# TODO  Double DQN
# TODO  replay buffer
# TODO  sample minibatches
# TODO  importance sample minibatches
# TODO  dueling dqn
# TODO  expire buffer

# s,a,r,s'
# a are ints from 0 to
# TODO set label of terminal state to just `r` instead of estimate of Q
# TODO make function to get labels for minibatch
# TODO target network
# TODO sample buffer in less naive way (currently just wait till it's full and train

env = gym.make('FrozenLake-v0')

S = env.observation_space.n
A = env.action_space.n
TERMINAL_STATE = S - 1

# Use `deque` because it's efficient to remove the leading elements to expire them.

BATCH_SIZE = 1024
epsilon = args.exploration_rate
gamma = args.discount
BUFFER_SIZE = args.buffer_size
ITERS = args.iterations

buffer = deque()


def eps_greedy(s: np.int64, epsilon=epsilon):
    # `np.argmax` works along flattened array, so for an nested array with a single entry, we get the right answer.
    return np.argmax(Q.predict(to_categorical(s, S))
                     ) if random.random() > epsilon else env.action_space.sample()


if __name__ == '__main__':
    Q = load_model('model.h5') if not args.new and os.path.exists('model.h5') else create_q(S, A)

    # initial sampling
    running_rews = []
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

        # All terminal states have 0 reward and themselves as a successor state for all actions.
        for a in range(A):
            buffer.append([TERMINAL_STATE, a, 0, TERMINAL_STATE])

        # get last reward as score for whole episode to see OpenAI score
        running_rews.append(r)
        if i % 10_000 == 0 and i > 0:
            print(np.mean(running_rews))
            running_rews.clear()

        if len(buffer) >= BUFFER_SIZE:
            data = np.array(buffer)

            # One-hot encode states.
            states = to_categorical(data[:, 0], num_classes=S).astype(np.float32)
            succ_states = to_categorical(data[:, 3], num_classes=S).astype(np.float32)

            # Actions must be ints so we can use them as indices.
            actions = data[:, 1].astype(np.int)
            rewards = data[:, 2].astype(np.float32)

            td_estimates = Q.predict(states)

            for td_estimate, a, r, s_ in zip(td_estimates, actions, rewards, succ_states):
                # `Q.predict` returns a (1,A) array, so we use [0] to extract the sub-array.
                td_estimates[a] += r + gamma * Q.predict(s_[None, :])[0][a]

            Q.fit(
                x=states,
                y=td_estimates,
                batch_size=BATCH_SIZE,
                validation_split=0.1,
                verbose=False,
            )

            # Empty replay buffer for next round of training.
            buffer.clear()

        if i % 5 == 0:
            if args.save:
                Q.save('model.h5')
