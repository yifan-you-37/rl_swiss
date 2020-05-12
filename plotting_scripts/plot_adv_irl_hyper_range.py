'''
check the heatmap for ant airl and fairl hyperparameters
'''
import joblib
import os
from os import path as osp
import numpy as np
import sys
sys.path.append('/data/rl_swiss')

import csv
from rlkit.core.vistools import plot_2dhistogram

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import seaborn as sns; sns.set()

import json

rews = [2.0]
gps = [2.0, 4.0, 8.0, 16.0]
rew_ind = {}
for i in range(len(rews)):
    rew_ind[rews[i]] = i
gp_ind = {}
for i in range(len(gps)):
    gp_ind[gps[i]] = i


def make_heatmap(grid, save_path, title):
    ax = sns.heatmap(grid, vmin=0, vmax=6500, cmap="YlGnBu")
    ax.set(xlabel='Gradient Penalty', ylabel='Reward Scale', xticklabels=gps, yticklabels=rews, title=title)
    ax.figure.savefig(save_path)
    plt.close()
def make_progress_plots(arr_progress, save_folder_path, title):
    for one_exp in arr_progress:
            y = one_exp['progress']['Test Returns Mean']
            x = np.ones((y.size)) * 100000
            x = np.cumsum(x)
            plt.plot(x, y, label='rew_{}_gp_{}'.format(one_exp['params']['sac_params']['reward_scale'], one_exp['params']['adv_irl_params']['grad_pen_weight']))
            plt.legend()
            plt.title(title)
    plt.savefig('{}/{}.png'.format(save_folder_path, title))
            
def extract_info_csv(csv_path):
    all_rows = []
    with open(csv_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                # print(f'Column names are {", ".join(row)}')
                line_count += 1
            line_count += 1
            all_rows.append(row)

    all_rows_dict = {k: np.array([float(dic[k]) for dic in all_rows]) for k in all_rows[0]}
    return all_rows_dict
def extract_info(exp_path):
    grid_ret = np.zeros((len(rews), len(gps)))
    arr_progress = []
    for d in os.listdir(exp_path):
        sub_path = osp.join(exp_path, d)
        with open(osp.join(sub_path, 'variant.json')) as f:
            json_dict = json.load(f)
            rew_scale = json_dict['sac_params']['reward_scale']
            gp_weight = json_dict['adv_irl_params']['grad_pen_weight']
        if gp_weight in gps and rew_scale in rews:
            test_ret = joblib.load(
                osp.join(sub_path, 'best.pkl')
            )['statistics']['Test Returns Mean']
            progress = extract_info_csv(osp.join(sub_path, 'progress.csv'))
            print(rew_scale, gp_weight, test_ret, rew_ind[rew_scale])
            grid_ret[rew_ind[rew_scale], gp_ind[gp_weight]] = test_ret
            arr_progress.append({
                'params': json_dict,
                'progress': progress
            })

    return grid_ret, arr_progress

if __name__ == '__main__':
    # extract the info for airl
    exp_path = '/data/rl_swiss/logs/airl-ant-hype-search'
    airl_grid_ret, airl_arr_progress = extract_info(exp_path)
    # make_heatmap(airl_grid_ret, '/data/rl_swiss/logs_plots/airl_hype_grid.png', '')
    make_progress_plots(airl_arr_progress, '/data/rl_swiss/logs_plots/', 'airl-ant-hype-search')

    # extract the info for fairl
    # exp_path = '/data/rl_swiss/logs/airl-ant-hype-search'
    # fairl_grid = extract_info(exp_path)
    # make_heatmap(fairl_grid, 'plots/junk_vis/fairl_hype_grid.png', '')
