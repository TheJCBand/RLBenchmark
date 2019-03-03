#/usr/bin/env python

import gym
import roboschool
import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.ddpg.policies import MlpPolicy as DDPGMlpPolicy
from stable_baselines.sac.policies import MlpPolicy as SACMlpPolicy
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C, DDPG, GAIL, PPO1, PPO2, SAC, TRPO, results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
import os
import glob
import csv
import matplotlib.pyplot as plt

total_timesteps = 1000000

envList = [spec.id for spec in gym.envs.registry.all() if spec.id.startswith('Roboschool')]
#envList = ['RoboschoolInvertedPendulum-v1']
algList = [A2C, DDPG, PPO1, PPO2, SAC, TRPO]
algNameList = ['A2C', 'DDPG', 'PPO1', 'PPO2', 'SAC', 'TRPO']

if not os.path.isdir('data'):
    os.mkdir('data')

for envName in envList:
    if not os.path.isdir('data/'+envName):
        os.mkdir('data/'+envName)
    for alg, algName in zip(algList, algNameList):
        env = Monitor(gym.make(envName), 'data/' + envName + '/' + algName + 'Log', True)
        env = DummyVecEnv([lambda: env])

        if alg == DDPG:
            n_actions = env.action_space.shape[-1]
            action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
            model = DDPG(DDPGMlpPolicy, env, action_noise=action_noise)
        elif alg == SAC:
            model = SAC(SACMlpPolicy, env)
        else:
            model = alg(MlpPolicy, env) 

        print('Training ' + algName + ' on ' + envName)
        model.learn(total_timesteps=total_timesteps)
        model.save('data/' + envName + '/' +  algName + 'Model')
#    results_plotter.plot_results('data/' + envName, total_timesteps, 'timesteps', envName)
#    if not os.path.isdir('data/plots'):
#        os.mkdir('data/plots')
#    plt.savefig('data/plots/' + envName + '.pdf')
