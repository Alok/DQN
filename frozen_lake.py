#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
import sys
from collections import deque

import gym
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import (Activation, BatchNormalization, Dense, Dropout,
                          Input, concatenate)
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical

# start with DQN, then Double DQN

# s,a,r,s'
# a are ints from 0 to
# TODO set label of terminal state to just `r` instead of estimate of Q
# TODO make function to get labels for minibatch
# TODO target network

env = gym.make('FrozenLake-v0')

S = env.observation_space.n
A = env.action_space.n

# Use `deque` because it's efficient to remove the leading elements to expire them.
buffer = deque()
batch_size = 64
epsilon = 0.99
gamma = 0.99
MAX_BUFFER_SIZE = 1_000_000
ITERS = 100

sess = tf.InteractiveSession()


def eps_greedy(s: np.int64, epsilon=epsilon):
    # `np.argmax` works along flattened array, so for an nested array with a single entry, we get the right answer.
    return np.argmax(Q.predict(to_categorical(s,S))) if random.random() > epsilon else env.action_space.sample()


if __name__ == '__main__':
    Q = create_q(S, A)

    # initial sampling
    for i in range(ITERS):

        epsilon *= .99

        while len(buffer) < MAX_BUFFER_SIZE:
            done = False
            # Decay exploration over time

            s = env.reset()

            while not done:
                a = eps_greedy(s)
                s_, r, done, _ = env.step(a)
                buffer.append([s, a, r, s_])
                s = s_

        data = np.array(buffer)

        # one hot encode states
        states = to_categorical(data[:, 0], num_classes=S).astype(np.float32)
        succ_states = to_categorical(data[:, 3], num_classes=S).astype(np.float32)

        actions = data[:, 1].astype(np.int)  # actions must be ints so we can use them as indices
        rewards = data[:, 2].astype(np.float32)

        td_estimates = Q.predict(states)

        for td_estimate, a, r, s_ in zip(td_estimates, actions, rewards, succ_states):
            # `Q.predict` returns a (1,A) array, so we use [0] to get the item out
            td_estimates[a] += r + gamma * Q.predict(s_[None, :])[0][a]

        Q.fit(states, td_estimates)

        # empty replay for next round
        buffer.clear()
