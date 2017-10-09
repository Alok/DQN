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
epsilon = 0.05
gamma = 0.99


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
        return tf.convert_to_tensor(to_categorical(x, num_classes=S))

    else:
        print(type(x), x)
    return tensor


def eps_greedy(s: tf.Tensor, epsilon=epsilon):
    return np.argmax(Q(s)) if random.random() > epsilon else env.action_space.sample()


if __name__ == '__main__':
    Q = create_q()
    # initial sampling
    for _ in range(100):
        done = False

        s = env.reset()

        # random sampling to just get some training data
        while not done:
            a = eps_greedy(T(s))
            s_, r, done, _ = env.step(a)
            buffer.append(np.array([s, a, r, s_]))
            s = s_

    train = np.random.permutation(buffer).astype(np.float32)

    mb = train[:batch_size]

# TODO set label of terminal state to just r instead of estimate of Q

# TODO make function to get labels for minibatch
# TODO target network
