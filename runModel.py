import gym
import roboschool
import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.ddpg.policies import MlpPolicy as DDPGMlpPolicy
from stable_baselines.sac.policies import MlpPolicy as SACMlpPolicy
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C, DDPG, GAIL, PPO1, PPO2, SAC, TRPO, results_plotter

envName = 'RoboschoolHopper-v1'
algName = 'PPO2'

algList = [A2C, DDPG, PPO1, PPO2, SAC, TRPO]
algNameList = ['A2C', 'DDPG', 'PPO1', 'PPO2', 'SAC', 'TRPO']
algDict = dict(zip(algNameList,algList))

env = DummyVecEnv([lambda: gym.make(envName)])
model = algDict[algName].load('data/'+envName+'/'+algName+'Model.pkl')
model.set_env(env)
observation = env.reset()
env.render(mode='human')
done = False
#while not done:
for i in range(1000):
    action, state = model.predict(observation)
    observation, reward, done, info = env.step(action)
