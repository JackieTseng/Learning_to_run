import opensim as osim
from osim.http.client import Client
from osim.env import *
import numpy as np
import argparse

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, concatenate
from keras.optimizers import Adam
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from keras.optimizers import RMSprop

import argparse
import math

# Settings
remote_base = 'http://grader.crowdai.org:1729'
crowdai_token = "68535f05c7d17429755a34fb2bbb30e0"

# Command line parameters
parser = argparse.ArgumentParser(description='Submit the result to crowdAI')
parser.add_argument('--token', dest='token', action='store')
parser.add_argument('--model', dest='model', action='store', default="sample.h5f")
args = parser.parse_args()

env = RunEnv(visualize=False)
client = Client(remote_base)
nb_actions = env.action_space.shape[0]

# Create environment
observation = client.env_create(crowdai_token)

# IMPLEMENTATION OF YOUR CONTROLLER
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(32))
actor.add(Activation('relu'))
actor.add(Dense(32))
actor.add(Activation('relu'))
actor.add(Dense(32))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('sigmoid'))

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = concatenate([action_input, flattened_observation])
x = Dense(64)(x)
x = Activation('relu')(x)
x = Dense(64)(x)
x = Activation('relu')(x)
x = Dense(64)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)

memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.2, size=env.noutput)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.99, target_model_update=1e-3,
                  delta_clip=1.)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])
agent.load_weights(args.model)

def my_controller(agent, observation):
    return agent.forward(observation)

# The grader runs 3 simulations of at most 1000 steps each. We stop after the last one
while True:
    #[observation, reward, done, info] = client.env_step(env.action_space.sample().tolist())
    [observation, reward, done, info] = client.env_step(my_controller(agent, observation).tolist(), True)
    if done:
        observation = client.env_reset()
        if not observation:
            break

client.submit()
