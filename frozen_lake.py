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
epsilon = 0.90
gamma = 0.99
ITERS = 1000

sess = tf.InteractiveSession()


def create_q():
    state = Input(shape=(S, ))
    # action = Input(shape=(A, ))
    #
    # x = concatenate([state, action])
    x = state

    x = Dense(16, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(16, activation='relu')(x)
    x = BatchNormalization()(x)

    rewards = Dense(A)(x)
    # rewards = Dense(1)(x)

    model = Model(inputs=state, outputs=rewards)
    # model = Model(inputs=[state, action], outputs=rewards)

    model.compile(
        optimizer=Adam(),
        loss='mse',
        metrics=['acc'],
    )

    return model


def eps_greedy(s: np.ndarray, epsilon=epsilon):
    return np.argmax(Q.predict(s)) if random.random() > epsilon else env.action_space.sample()


if __name__ == '__main__':
    Q = create_q()
    # initial sampling
    for i in range(ITERS):
        done = False
        # decay exploration over time
        if i % (ITERS // 10) == 0 and i > 0:
            epsilon *= .95

        s = env.reset()

        # random sampling to just get some training data
        while not done:
            a = eps_greedy(to_categorical(s, num_classes=S))
            s_, r, done, _ = env.step(a)
            # TODO encode buffer without one hot encoding everything
            # buffer.append(([to_categorical(s), to_categorical(a), np.array(r), to_categorical(s)]))
            buffer.append([s, a, r, s_])
            s = s_

    # XXX can the array be shuffled as long as we store the succ state and current reward?

    data = np.array(buffer)

    # one hot encode states
    states = to_categorical(data[:, 0], num_classes=S).astype(np.float32)
    succ_states = to_categorical(data[:, 3], num_classes=S).astype(np.float32)
    actions = data[:, 1].astype(np.int)
    # actions = to_categorical(data[:, 1]).astype(np.float32)
    rewards = data[:, 2].astype(np.float32)

    # td_estimate = rewards + gamma * Q.predict(succ_states, actions)
    td_estimates = Q.predict(states)

    for i, (s, a, r, s_) in enumerate(zip(states, actions, rewards, succ_states)):
        # `Q.predict` returns a (1,A) array
        td_estimates[i][a] += r + gamma * Q.predict(s_[None, :])[0][a]

    # td_estimate = rewards + gamma * Q.predict(succ_states, actions)
    Q.fit(states, td_estimates)
