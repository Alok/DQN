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
