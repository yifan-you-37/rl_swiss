import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from copy import deepcopy

from gym.spaces import Dict

from rllab.misc.instrument import VariantGenerator
import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger, set_seed

from rlkit.envs import get_meta_env, get_meta_env_params_iters
from rlkit.envs.wrappers import ScaledMetaEnv

from rlkit.torch.sac.policies import ReparamTanhMultivariateGaussianPolicy
from rlkit.torch.networks import Mlp

from rlkit.torch.irl.fetch_few_shot_lin_classification_task_design import FetchLinClassTaskDesign
from rlkit.torch.irl.encoders.mlp_encoder import TimestepBasedEncoder, WeightShareTimestepBasedEncoder
from rlkit.torch.irl.encoders.conv_seq_encoder import ConvTrajEncoder, R2ZMap, Dc2RMap, NPEncoder

import yaml
import argparse
import importlib
import psutil
import os
from os import path
import argparse
import joblib
from time import sleep


def experiment(variant):
    # make the disc model
    z_dim = variant['algo_params']['z_dim']

    # make the MLP
    # hidden_sizes = [variant['algo_params']['mlp_hid_dim']] * variant['algo_params']['mlp_layers']
    # mlp = Mlp(
    #     hidden_sizes,
    #     output_size=1,
    #     input_size=48 + z_dim,
    #     batch_norm=variant['algo_params']['mlp_use_bn']
    # )

    algorithm = FetchLinClassTaskDesign(
        # mlp,
        **variant['algo_params']
    )

    # for _ in range(100):
    #     # print(algorithm._get_any())
    #     # print(algorithm._get_except(0,1))

    #     img = algorithm._get_image_without_object(1, 2)
    #     print('-------')
    #     print(img[:6])
    #     print(img[6:12])
    #     print(img[12:18])
    #     print(img[18:24])
    # 1/0

    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()

    return 1


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', help='experiment specification file')
    args = parser.parse_args()
    with open(args.experiment, 'r') as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string)
    
    if exp_specs['use_gpu']:
        print('\n\nUSING GPU\n\n')
        ptu.set_gpu_mode(True)
    exp_id = exp_specs['exp_id']
    exp_prefix = exp_specs['exp_name']
    seed = exp_specs['seed']
    set_seed(seed)
    setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs)

    experiment(exp_specs)
