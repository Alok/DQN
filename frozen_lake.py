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
# TODO make Q output an array of rewards for each action

env = gym.make('FrozenLake-v0')

S = env.observation_space.n
A = env.action_space.n

# Use deque because it's easy to remove the leading elements to expire them.
buffer = deque()
batch_size = 64
epsilon = 0.90
gamma = 0.99
ITERS = 10

sess = tf.InteractiveSession()


def create_q():
    state = Input(shape=(S, ))

    # action = Input(shape=A)
    #
    # x = concatenate([state, action])
    x = state

    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)

    rewards = Dense(A)(x)

    model = Model(inputs=state, outputs=rewards)

    model.compile(
        optimizer=Adam(),
        loss='mse',
        metrics=['acc'],
    )

    return model


def T(x) -> tf.Tensor:
    if isinstance(x, np.ndarray):
        tensor = tf.convert_to_tensor(x, dtype=tf.float32)
    elif isinstance(x, tf.Tensor):
        tensor = x
    elif isinstance(x, list):
        tensor = tf.convert_to_tensor(x, dtype=tf.float32)
    elif isinstance(x, (np.int64, np.int32, np.int)):
        return tf.convert_to_tensor(to_categorical(x, num_classes=S).astype(np.float32))
    else:
        print(type(x), x)
    return tensor


def eps_greedy(s: tf.Tensor, epsilon=epsilon):
    return np.argmax(Q(s)) if random.random() > epsilon else env.action_space.sample()


if __name__ == '__main__':
    Q = create_q()
    # initial sampling
    for i in range(ITERS):
        done = False
        if i % (ITERS // 10) == 0 and i > 0:
            epsilon *= .95

        s = env.reset()

        # random sampling to just get some training data
        while not done:
            a = eps_greedy(T(s))
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

    rewards = data[:, 2]
    # TODO is this masking correct? the reduce sum seems to be a cheap hack for fancy indexing
    action_mask = T(to_categorical(actions, num_classes=A))
    td_estimate = rewards + gamma * tf.reduce_sum(action_mask * Q(T(succ_states)), axis=1)

# TODO use Q.predict and drop the tensors, and use fancy indexing from numpy

    td_estimate = rewards + gamma * tf.reduce_sum(action_mask * Q(T(succ_states)), axis=1)

    Q.fit(states, td_estimate)

# TODO set label of terminal state to just `r` instead of estimate of Q

# TODO make function to get labels for minibatch
# TODO target network
# TODO add sess.close or `with tf.session as sess`
