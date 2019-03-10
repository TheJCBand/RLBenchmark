from baselines.common import plot_util as pu
import matplotlib.pyplot as plt
import numpy as np

dataDir = 'baselinesData'
envName = 'HalfCheetah-v2'
alg = 'ppo2'
results = pu.load_results(dataDir+'/'+envName)
#r = results[0]
#plt.plot(np.cumsum(r.monitor.l), r.monitor.r)
pu.plot_results(results)
plt.show()
