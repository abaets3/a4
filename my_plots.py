### COPY AND PASTED FROM BETTERMDPTOOLS GITHUB ###
### WITH A MAJOR MODIFICATIONS BY ME ###



import math

import gym
import pygame
from algorithms.rl import RL
from algorithms.planner import Planner
from examples.test_env import TestEnv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from matplotlib.colors import LinearSegmentedColormap


class my_plots:
    @staticmethod
    def grid_world_policy_plot_cliff(data, label, size, title):
            cliff = np.zeros(size, dtype=bool)
            cliff[3, 1:-1] = True
            data = np.around(np.array(data).reshape(size), 2)
            data = data + 2

            for i in range(len(data)):
                for j in range(len(data[0])):
                    if cliff[i][j]:
                        data[i][j] = 1
            
            data[len(data)-1][len(data[0])-1] = 0

            df = pd.DataFrame(data=data)
            my_colors = ((0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0), (0.8, 0.0, 0.0, 1.0), (0.8, 0.8, 0.0, 1.0), (0.0, 0.0, 0.8, 1.0), (0.8, 0.0, 0.8, 1.0))
            cmap = LinearSegmentedColormap.from_list('Custom', my_colors, len(my_colors))
            ax = sns.heatmap(df, cmap=cmap, linewidths=1.0)
            colorbar = ax.collections[0].colorbar
            colorbar.set_ticks([.33, 1, 1.66667, 2.333333, 3, 3.6777])
            colorbar.set_ticklabels(['Goal', 'Cliff', 'Left', 'Up', 'Right', 'Down'])
            plt.title(label)
            plt.savefig(f'plots/{title}.png')
            plt.close()

    @staticmethod
    def grid_values_heat_map_cliff(data, label, size, title):
            cliff = np.zeros(size, dtype=bool)
            cliff[3, 1:-1] = True
            data = np.around(np.array(data).reshape(size), 2)
            #data = data + 2

            for i in range(len(data)):
                for j in range(len(data[0])):
                    if cliff[i][j]:
                        data[i][j] = 0
            
            #data[len(data)-1][len(data[0])-1] = 1
            data = np.around(np.array(data).reshape(size), 2)
            df = pd.DataFrame(data=data)
            sns.heatmap(df, annot=False).set_title(label)
            plt.savefig(f'plots/{title}.png')
            plt.close()

    @staticmethod
    def grid_world_policy_plot_frozen(data, label, size, desc, title):
            data = np.around(np.array(data).reshape(size), 2)
            data = data + 2

            for i in range(len(data)):
                for j in range(len(data[0])):
                    if desc[i][j] == b'H':
                        data[i][j] = 1
                    if desc[i][j] == b'G':
                        data[i][j] = 0

            df = pd.DataFrame(data=data)
            my_colors = ((0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0), (0.8, 0.0, 0.0, 1.0), (0.8, 0.8, 0.0, 1.0), (0.0, 0.0, 0.8, 1.0), (0.8, 0.0, 0.8, 1.0))
            cmap = LinearSegmentedColormap.from_list('Custom', my_colors, len(my_colors))
            ax = sns.heatmap(df, cmap=cmap, linewidths=1.0)
            colorbar = ax.collections[0].colorbar
            labels = ['Goal', 'Holes', 'Left', 'Down', 'Right', 'Up']
            labels_nums = np.arange(0,len(labels),1)
            loc = labels_nums
            colorbar.set_ticks(loc) # https://stackoverflow.com/questions/18704353/correcting-matplotlib-colorbar-ticks
            colorbar.set_ticklabels(labels)
            plt.title(label)
            plt.savefig(f'plots/{title}.png')
            plt.close()

    @staticmethod
    def grid_values_heat_map_frozen(data, label, size, desc, title):
            data = np.around(np.array(data).reshape(size), 2)

            for i in range(len(data)):
                for j in range(len(data[0])):
                    if desc[i][j] == b'H':
                        data[i][j] = 0
                    if desc[i][j] == b'G':
                        data[i][j] = 10
            
            data = np.around(np.array(data).reshape(size), 2)
            df = pd.DataFrame(data=data)
            sns.heatmap(df, annot=False).set_title(label)
            plt.savefig(f'plots/{title}.png')
            plt.close()

    @staticmethod
    def subplotted_grid_values_heat_map_frozen(data, label, labels, size, desc, title, actual_title):
        plt.suptitle(actual_title)
        for i in range(len(data)):
            current_data = data[i]
            current_data = np.around(np.array(current_data).reshape(size), 2)

            for k in range(len(current_data)):
                for j in range(len(current_data[0])):
                    if desc[i][k][j] == b'H':
                        current_data[k][j] = 0
                    if desc[i][k][j] == b'G':
                        current_data[k][j] = 10
            
            current_data = np.around(np.array(current_data).reshape(size), 2)
            df = pd.DataFrame(data=current_data)
            plt.subplot(2,2,i+1)
            sns.heatmap(df, annot=False, xticklabels=False, yticklabels=False).set_title(f"{label} of {labels[i]}")
        plt.savefig(f'plots/{title}.png')
        plt.close()

    @staticmethod
    def subplotted_grid_values_heat_map_cliffs(data, label, labels, size, title, actual_title):
        plt.suptitle(actual_title)
        for i in range(len(data)):
            current_data = data[i]
            cliff = np.zeros(size, dtype=bool)
            cliff[3, 1:-1] = True
            current_data = np.around(np.array(current_data).reshape(size), 2)

            for k in range(len(current_data)):
                for j in range(len(current_data[0])):
                    if cliff[k][j]:
                        current_data[k][j] = 0
            
            current_data = np.around(np.array(current_data).reshape(size), 2)
            df = pd.DataFrame(data=current_data)
            plt.subplot(2,2,i+1)
            sns.heatmap(df, annot=False, xticklabels=False, yticklabels=False).set_title(f"{label} of {labels[i]}")
        plt.savefig(f'plots/{title}.png')
        plt.close()

    @staticmethod
    def v_iters_plot(data, label):
        df = pd.DataFrame(data=data)
        df.columns = [label]
        sns.set_theme(style="whitegrid")
        title = label + " v Iterations"
        sns.lineplot(x=df.index, y=label, data=df).set_title(title)
        plt.show()

if __name__ == "__main__":
    frozen_lake = gym.make('FrozenLake8x8-v1', render_mode=None)

    # VI/PI grid_world_policy_plot
    # V, V_track, pi = Planner(frozen_lake.env.P).value_iteration()
    # n_states = frozen_lake.env.observation_space.n
    # new_pi = list(map(lambda x: pi(x), range(n_states)))
    # s = int(math.sqrt(n_states))
    # Plots.grid_world_policy_plot(np.array(new_pi), "Grid World Policy")

    # Q-learning grid_world_policy_plot
    # Q, V, pi, Q_track, pi_track = RL(frozen_lake.env).q_learning()
    # n_states = frozen_lake.env.observation_space.n
    # new_pi = list(map(lambda x: pi(x), range(n_states)))
    # s = int(math.sqrt(n_states))
    # Plots.grid_world_policy_plot(np.array(new_pi), "Grid World Policy")

    # Q-learning v_iters_plot
    # Q, V, pi, Q_track, pi_track = RL(frozen_lake.env).q_learning()
    # max_reward_per_iter = np.amax(np.amax(Q_track, axis=2), axis=1)
    # Plots.v_iters_plot(max_reward_per_iter, "Reward")

    # VI/PI v_iters_plot
    # V, V_track, pi = Planner(frozen_lake.env.P).value_iteration()
    # V, V_track, pi = Planner(frozen_lake.env.P).policy_iteration()
    # max_value_per_iter = np.amax(V_track, axis=1)
    # Plots.v_iters_plot(max_value_per_iter, "Value")

    # Q-learning grid_values_heat_map
    # Q, V, pi, Q_track, pi_track = RL(frozen_lake.env).q_learning()
    # Plots.grid_values_heat_map(V, "State Values")

    # VI/PI grid_values_heat_map
    V, V_track, pi = Planner(frozen_lake.env.P).value_iteration()
    V, V_track, pi = Planner(frozen_lake.env.P).policy_iteration()
    Plots.grid_values_heat_map(V, "State Values")