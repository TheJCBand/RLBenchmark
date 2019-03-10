#!/bin/bash
envList=$(python getMujocoEnvList.py)
declare -a algList=("a2c" "ddpg" "gail" "ppo2" "trpo")
numTimesteps=1000000

rootDir=$HOME/rLStuff/RLBenchmark
dataDir="baselinesData"
for env in $envList
do
    mkdir $rootDir/$dataDir/$env/
    savePath="$rootDir/$dataDir/$env/"
    for alg in "${algList[0]}"
    do
        OPENAI_LOGDIR=$rootDir/$dataDir/$env OPENAI_LOG_FOMAT=csv python -m baselines.run --env=$env --alg=$alg --num_timesteps=$numTimesteps
    done
done
