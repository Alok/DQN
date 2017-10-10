#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import keras
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import (Activation, BatchNormalization, Dense, Dropout,
                          Input, concatenate)
from keras.models import Model, Sequential
from keras.optimizers import Adam


def create_q(S, A):
    state = Input(shape=(S, ))
    x = state

    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)

    rewards = Dense(A, activation='linear')(x)

    model = Model(inputs=state, outputs=rewards)

    model.compile(
        optimizer=Adam(),
        loss='mse',
        metrics=['acc'],
    )

    return model
