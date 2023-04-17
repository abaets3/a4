import gym
import pygame
from algorithms.rl import RL
from examples.test_env import TestEnv
from algorithms.planner import Planner
from my_planner import my_planner
from examples.plots import Plots
import my_frozen_lake
import my_taxi
import my_cliff_walking
from my_plots import my_plots
from my_frozen_lake import generate_random_map
import numpy as np
import math
import matplotlib.pyplot as plt
import time

import warnings
warnings.filterwarnings("ignore")


def CliffWalkingExperiments():
    # Make original problem
    cliff = gym.make('MyCliffWalking', render_mode=None, size=12, goal_reward=10, normal_reward=-1, hazard_reward=-10)
    V, V_track, pi = my_planner(cliff.env.P).value_iteration(theta=1e-10, n_iters=100)

    # Get the heat map
    my_plots.grid_values_heat_map_cliff(V, "State Values for Cliff Walking", size=(4,12), title="CliffWalkingSize12HeatMap")

    # Get the policy
    n_states = cliff.env.observation_space.n
    new_pi = list(map(lambda x: pi(x), range(n_states)))
    my_plots.grid_world_policy_plot_cliff(np.array(new_pi), "Cliff Walking Ideal Policy", size=(4,12), title="CliffWalkingSize12Policy")

    # Start on Value iteration
    times = []
    for i in [64, 128, 256, 512]:
        data = []
        iters = 0
        cur_time = 0
        for j in [.7, .8, .9, 1]:
            cliff = gym.make('MyCliffWalking', render_mode=None, size=int(i/4), goal_reward=100, normal_reward=-.001, hazard_reward=-10)
            start = time.time()
            V, V_track, pi = my_planner(cliff.env.P).value_iteration(theta=1e-20, n_iters=200, gamma = j)
            mean = np.trim_zeros(np.mean(V_track, axis=1))
            normalized_mean = (mean/(mean[-1]))
            cur_time = cur_time + time.time()-start
            iters = iters + len(normalized_mean)
            plt.plot(normalized_mean, label=f"gamma of {j}")
            if data == []:
                data = V
            else:
                data = np.vstack((data, V))
        plt.title(f"Cliff Walking Value Iteration Convergence with Problem Size of {i}")
        plt.ylabel("Normalized mean")
        plt.xlabel("Iteration")
        plt.legend(loc="lower right")
        plt.savefig(f"plots/CliffWalkingValueIterationSize{i}.png")
        plt.close()

        times.append(cur_time/iters)

        my_plots.subplotted_grid_values_heat_map_cliffs(data, "Gamma", [.7, .8, .9, 1], size=(4,int(i/4)), title=f"CliffWalkingSize{i}VSubHeatMaps", actual_title=f"Cliff Walking Value Iteration Heat Maps for Size {i}")

    plt.plot([64, 128, 256, 512], times)
    plt.title(f"Cliff Walking Value Iteration Time")
    plt.ylabel("Time per iteration")
    plt.xlabel("Problem Size")
    plt.savefig(f"plots/CliffWalkingValueIterationTime.png")
    plt.close()

    # Then on Policy iteration
    times = []
    for i in [64, 128, 256, 512]:
        data = []
        iters = 0
        cur_time = 0
        for j in [.7, .8, .9, 1]:
            cliff = gym.make('MyCliffWalking', render_mode=None, size=int(i/4), goal_reward=100, normal_reward=-.001, hazard_reward=-10)
            V, V_track, pi = my_planner(cliff.env.P).policy_iteration(theta=1e-20, n_iters=200, gamma = j)
            mean = np.trim_zeros(np.mean(V_track, axis=1))
            start = time.time()
            normalized_mean = (mean/(mean[-1]))
            cur_time = cur_time + time.time()-start
            iters = iters + len(normalized_mean)
            plt.plot(normalized_mean, label=f"gamma of {j}")
            if data == []:
                data = V
            else:
                data = np.vstack((data, V))
        plt.title(f"Cliff Walking Policy Iteration Convergence with Problem Size of {i}")
        plt.ylabel("Normalized mean")
        plt.xlabel("Iteration")
        plt.legend(loc="lower right")
        plt.savefig(f"plots/CliffWalkingPolicyIterationSize{i}.png")
        plt.close()

        times.append(cur_time/iters)

        my_plots.subplotted_grid_values_heat_map_cliffs(data, "Gamma", [.7, .8, .9, 1], size=(4,int(i/4)), title=f"CliffWalkingSize{i}PSubHeatMaps", actual_title=f"Cliff Walking Policy Iteration Heat Maps for Size {i}")
    
    plt.plot([64, 128, 256, 512], times)
    plt.title(f"Cliff Walking Policy Iteration Time")
    plt.ylabel("Time per iteration")
    plt.xlabel("Problem Size")
    plt.savefig(f"plots/CliffWalkingPolicyIterationTime.png")
    plt.close()

def FrozenLakeExperiments():

    # Make original problem
    frozen_lake = gym.make('MyFrozenLake', render_mode=None, size=4, hole_probability=.85, is_slippery=True, slippery_probability=.1, hazard_reward=-1, goal_reward=10, normal_reward=-.1)
    V, V_track, pi = my_planner(frozen_lake.env.P).value_iteration(theta=1e-10, n_iters=100)

    # Get the heat map
    my_plots.grid_values_heat_map_frozen(V, "State Values for Frozen Lake", size=(4,4), desc=frozen_lake.desc, title="FrozenLakeSize64HeatMap")

    # Get the policy
    n_states = frozen_lake.env.observation_space.n
    new_pi = list(map(lambda x: pi(x), range(n_states)))
    my_plots.grid_world_policy_plot_frozen(np.array(new_pi), "Frozen Lake Ideal Policy", size=(4,4), desc=frozen_lake.desc, title="FrozenLakeSize64Policy")

    # Start on Value iteration
    times = []
    for i in [64, 128, 256, 512]:
        data = []
        recorded_desc = []
        iters = 0
        cur_time = 0
        for j in [.7, .8, .9, 1]:
            frozen_lake = gym.make('MyFrozenLake', render_mode=None, size=int(math.sqrt(i)), hole_probability=.85, is_slippery=True, slippery_probability=.1, hazard_reward=-1, goal_reward=100, normal_reward=-.01)
            start = time.time()
            V, V_track, pi = my_planner(frozen_lake.env.P).value_iteration(theta=1e-20, n_iters=100, gamma = j)
            mean = np.trim_zeros(np.mean(V_track, axis=1))
            normalized_mean = (mean/(mean[-1]))
            plt.plot(normalized_mean, label=f"gamma of {j}")
            cur_time = cur_time + time.time()-start
            iters = iters + len(normalized_mean)
            if data == []:
                data = V
                recorded_desc = [frozen_lake.desc]
            else:
                data = np.vstack((data, V))
                recorded_desc = np.concatenate((recorded_desc, [frozen_lake.desc]))

        plt.title(f"Frozen Lake Value Iteration Convergence with Problem Size of {i}")
        plt.ylabel("Normalized mean")
        plt.xlabel("Iteration")
        plt.legend(loc="lower right")
        plt.savefig(f"plots/FrozenLakeValueIterationSize{i}.png")
        plt.close()

        times.append(cur_time/iters)

        my_plots.subplotted_grid_values_heat_map_frozen(data, "Gamma", [.7, .8, .9, 1], size=(int(math.sqrt(i)), int(math.sqrt(i))), desc=recorded_desc, title=f"FrozenLakeSize{i}VSubHeatMaps", actual_title=f"Frozen Lake Value Iteration Heat Maps for Size {i}")

    plt.plot([64, 128, 256, 512], times)
    plt.title(f"Frozen Lake Value Iteration Time")
    plt.ylabel("Time per iteration")
    plt.xlabel("Problem Size")
    plt.savefig(f"plots/FrozenLakeValueIterationTime.png")
    plt.close()

    # Up next, Policy iteration
    times = []
    for i in [64, 128, 256, 512]:
        data = []
        recorded_desc = []
        iters = 0
        cur_time = 0
        for j in [.7, .8, .9, 1]:
            frozen_lake = gym.make('MyFrozenLake', render_mode=None, size=int(math.sqrt(i)), hole_probability=.85, is_slippery=True, slippery_probability=.1, hazard_reward=-1, goal_reward=100, normal_reward=-.01)
            start = time.time()
            V, V_track, pi = my_planner(frozen_lake.env.P).policy_iteration(theta=1e-20, n_iters=100, gamma = j)
            mean = np.trim_zeros(np.mean(V_track, axis=1))
            normalized_mean = (mean/(mean[-1]))
            plt.plot(normalized_mean, label=f"gamma of {j}")
            cur_time = cur_time + time.time()-start
            iters = iters + len(normalized_mean)
            if data == []:
                data = V
                recorded_desc = [frozen_lake.desc]
            else:
                data = np.vstack((data, V))
                recorded_desc = np.concatenate((recorded_desc, [frozen_lake.desc]))

        plt.title(f"Frozen Lake Policy Iteration Convergence with Problem Size of {i}")
        plt.ylabel("Normalized mean")
        plt.xlabel("Iteration")
        plt.legend(loc="lower right")
        plt.savefig(f"plots/FrozenLakePolicyIterationSize{i}.png")
        plt.close()

        times.append(cur_time/iters)

        my_plots.subplotted_grid_values_heat_map_frozen(data, "Gamma", [.7, .8, .9, 1], size=(int(math.sqrt(i)), int(math.sqrt(i))), desc=recorded_desc, title=f"FrozenLakeSize{i}PSubHeatMaps", actual_title=f"Frozen Lake Policy Iteration Heat Maps for Size {i}")

    plt.plot([64, 128, 256, 512], times)
    plt.title(f"Frozen Lake Policy Iteration Time")
    plt.ylabel("Time per iteration")
    plt.xlabel("Problem Size")
    plt.savefig(f"plots/FrozenLakePolicyIterationTime.png")
    plt.close()

def CliffWalkingQExperiments():
    times = []
    for i in [64, 128, 256, 512]:
        data = []
        iters = 0
        cur_time = 0
        for j in [.25, .3, .35, .40]:
            cliff = gym.make('MyCliffWalking', render_mode=None, size=int(i/4), goal_reward=100, normal_reward=-.001, hazard_reward=-10)
            start = time.time()
            Q, V, pi, Q_track, pi_track = RL(cliff.env).q_learning(init_epsilon=j, n_episodes = i*10)
            mean = np.trim_zeros(np.mean(np.amax(Q_track, axis=2), axis=1)) # https://edstem.org/us/courses/32923/discussion/2947873
            normalized_mean = (mean/(mean[-1]))
            cur_time = cur_time + time.time()-start
            iters = iters + len(normalized_mean)
            plt.plot(normalized_mean, label=f"epislon of {j}")
            if data == []:
                data = V
            else:
                data = np.vstack((data, V))
        plt.title(f"Cliff Walking Q Learning Convergence with Problem Size of {i}")
        plt.ylabel("Normalized mean")
        plt.xlabel("Iteration")
        plt.legend(loc="lower right")
        plt.savefig(f"plots/CliffWalkingQLearningSize{i}.png")
        plt.close()

        times.append(cur_time/iters)
        
        my_plots.subplotted_grid_values_heat_map_cliffs(data, "Epsilon", [.25, .3, .35, .40], size=(4,int(i/4)), title=f"CliffWalkingSize{i}QSubHeatMaps", actual_title=f"Cliff Walking Q learning Heat Maps for Size {i}")

    plt.plot([64, 128, 256, 512], times)
    plt.title(f"Cliff Walking Q Learning Time")
    plt.ylabel("Time per iteration")
    plt.xlabel("Problem Size")
    plt.savefig(f"plots/CliffWalkingQLearningTime.png")
    plt.close()

def FrozenLakeQExperiments():
    times = []
    for i in [64, 128, 256, 512]:
        data = []
        recorded_desc = []
        iters = 0
        cur_time = 0
        for j in [.4, .6, .8, 1]:
            frozen_lake = gym.make('MyFrozenLake', render_mode=None, size=int(math.sqrt(i)), hole_probability=.85, is_slippery=True, slippery_probability=.1, hazard_reward=-1, goal_reward=10, normal_reward=-.01)
            start = time.time()
            Q, V, pi, Q_track, pi_track = RL(frozen_lake.env).q_learning(init_epsilon=j, n_episodes = i*100)
            mean = np.trim_zeros(np.mean(np.amax(Q_track, axis=2), axis=1)) # https://edstem.org/us/courses/32923/discussion/2947873
            normalized_mean = (mean/(mean[-1]))
            cur_time = cur_time + time.time()-start
            iters = iters + len(normalized_mean)
            plt.plot(normalized_mean, label=f"epislon of {j}")
            if data == []:
                data = V
                recorded_desc = [frozen_lake.desc]
            else:
                data = np.vstack((data, V))
                recorded_desc = np.concatenate((recorded_desc, [frozen_lake.desc]))
        plt.title(f"Frozen Lake Q Learning Convergence with Problem Size of {i}")
        plt.ylabel("Normalized mean")
        plt.xlabel("Iteration")
        plt.legend(loc="lower right")
        plt.savefig(f"plots/FrozenLakeQLearningSize{i}.png")
        plt.close()

        times.append(cur_time/iters)

        my_plots.subplotted_grid_values_heat_map_frozen(data, "Epsilon", [.4, .6, .8, 1], size=(int(math.sqrt(i)), int(math.sqrt(i))), desc=recorded_desc, title=f"FrozenLakeSize{i}QSubHeatMaps", actual_title=f"Frozen Lake Q learning Heat Maps for Size {i}")

    plt.plot([64, 128, 256, 512], times)
    plt.title(f"Frozen Lake Q Learning Time")
    plt.ylabel("Time per iteration")
    plt.xlabel("Problem Size")
    plt.savefig(f"plots/FrozenLakeQLearningTime.png")
    plt.close()

def QHyperTuning():
    cliff = gym.make('MyCliffWalking', render_mode=None, size=int(512/4), goal_reward=100, normal_reward=-.001, hazard_reward=-10)
    Q, V, pi, Q_track, pi_track = RL(cliff.env).q_learning(init_epsilon=.3, n_episodes = 5000, min_epsilon=0.3)

    # Get the heat map
    my_plots.grid_values_heat_map_cliff(V, "State Values for Cliff Walking for Epsilon=.3", size=(4,128), title="CliffWalkingSize512QLearningHeatMap")


print("Running Cliff Walking VI/PI")
CliffWalkingExperiments()
print("Running Frozen Lake VI/PI")
FrozenLakeExperiments()
print("Running Frozen Lake Q Learning")
FrozenLakeQExperiments()
print("Running Cliff Walking Q Learning")
CliffWalkingQExperiments()
print("Running Hyper Tuning Experiment")
QHyperTuning()