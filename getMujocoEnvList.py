from baselines import run 
import gym 
 
allEnvList = list(gym.envs.registry.all()) 
mujocoEnvList = [env.id for env in allEnvList if env._entry_point.startswith('gym.envs.mujoco')] 
returnStr = ''
for env in mujocoEnvList:
    returnStr += str(env) + ' '
print(returnStr)
