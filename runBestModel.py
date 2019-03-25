# Demo with best model from training

import deepQLearningSimple as dql
import gym
from keras.models import load_model 
from gym.envs.classic_control.mountain_car import MountainCarEnv
from gym.wrappers.time_limit import TimeLimit

import json



def run_demo(episode, n):
    """
    Run a demo for a given success episode from version v1.
    """
    print("="*60)
    print("Init success model from training episode %s" % episode)
    print("="*60)

    #model = "success_model_episode_318.h5"
    model = "success_model_episode_%s.h5"% episode

    pathImp = "data/model/version_simple_v1/"
    action_model = load_model(pathImp + model)

    # load new agent
    env = dql.patientMountainCar()
    agent = dql.agent(env  = env, training = False, render = True)

    # Implement the action model
    agent.action_dqn.dqn = action_model

    # Run
    agent.run(num_episode = n, num_steps = 200)
    env.close()


if __name__ == "__main__":

    run_demo(episode = 318, n = 10)
    
    print("DONE")
