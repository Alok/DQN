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
TERMINAL_STATE = S - 1  # XXX env specific

# Use `deque` because it's efficient to remove the leading elements to expire them.

BATCH_SIZE = 1024
epsilon = args.exploration_rate
gamma = args.discount
BUFFER_SIZE = args.buffer_size
TRAIN_SIZE = 100 * BATCH_SIZE
ITERS = args.iterations

# automatically handles expiring elements
buffer = deque(maxlen=BUFFER_SIZE)


def eps_greedy(s: np.int64, epsilon=epsilon):
    # `np.argmax` works along flattened array, so for an nested array with a single entry, we get the right answer.
    return np.argmax(Q.predict(to_categorical(s, S))
                     ) if random.random() > epsilon else env.action_space.sample()


if __name__ == '__main__':
    Q = load_model('target.h5') if not args.new and os.path.exists('target.h5') else create_q(S, A)
    target = clone_model(Q)

    # initial sampling
    running_rews = []  # XXX env specific
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
        # XXX env specific
        for a in range(A):
            buffer.append([TERMINAL_STATE, a, 0, TERMINAL_STATE])

        # XXX env specific
        # get last reward as score for whole episode to see OpenAI score
        running_rews.append(r)

        # XXX env specific
        if i % 1_000 == 0:
            successes = sum(1 for r in running_rews if r > 0)
            print(f'Score: {successes}/1000')
            running_rews.clear()

        if i % 1000 == 0 and len(buffer) >= BUFFER_SIZE:
            # sample from buffer
            data = np.array(random.sample(buffer, k=TRAIN_SIZE))

            # One-hot encode states.
            states = to_categorical(data[:, 0], num_classes=S).astype(np.float32)
            succ_states = to_categorical(data[:, 3], num_classes=S).astype(np.float32)

            # Actions must be ints so we can use them as indices.
            actions = data[:, 1].astype(np.int)
            rewards = data[:, 2].astype(np.float32)

            td_estimates = target.predict(states)

            for td_estimate, a, r, s_ in zip(td_estimates, actions, rewards, succ_states):
                # `Q.predict` returns a (1,A) array, so we use [0] to extract the sub-array.
                td_estimates[a] += r + gamma * np.max(target.predict(s_[None, :])[0])

            Q.fit(
                x=states,
                y=td_estimates,
                batch_size=BATCH_SIZE,
                validation_split=0.1,
                verbose=False,
            )

        if i % 10_000 and i > 0:
            # copy Q to update target
            target = clone_model(Q)

        if i % 10_000 == 0 and i > 0:
            if args.save:
                Q.save('model.h5')
                target.save('target.h5')
