import numpy as np
from policy_origin import Policy
from MDP import Environment
import torch.optim as optim
import torch
from replay_simu import ReplayMemory
import os
import sys

sys.path.append('../IRecGAN')
from util import get_args, get_optimizer


if __name__ == '__main__':
    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    outputdir = "output"
    # Define the environment
    num_clicks = 100
    recom_number = 20
    args = get_args()
    bsize = args.batch_size
    embed_dim = args.embed_dim
    encod_dim = args.nhid
    embed_dim_policy = args.embed_dim_policy
    encod_dim_policy = args.nhid_policy
    init_embed = args.init_embed
    model_type = args.model
    seedkey = args.seed
    optim = args.optim
    load = args.load

    if load:
        # Absolute route
        environment = "output/environment.pickle"
        policy_new = "output/agent.pickle"
        env = torch.load(environment)
        policy = torch.load(policy_new)
    else:
        env = Environment(bsize, embed_dim, encod_dim, num_clicks).to(device)
        policy = Policy(bsize, embed_dim_policy, encod_dim_policy, num_clicks - 1, recom_number).to(device)
        env.init_params()
        torch.save(env, os.path.join(outputdir, 'environment.pickle'))
        # Set initial policy
        policy.init_params()
        torch.save(policy, os.path.join(outputdir, 'orig_policy.pickle'))

    # Generate action and reward sequences
    capacity = 10000
    max_length = 5
    # Absolute route
    file_action = 'temp_data/gen_click.txt'
    file_reward = 'temp_data/gen_reward.txt'
    file_recom = 'temp_data/gen_action.txt'
    # for i in range(5):
    if os.path.isfile(file_action):
        os.remove(file_action)
    if os.path.isfile(file_reward):
        os.remove(file_reward)
    if os.path.isfile(file_recom):
        os.remove(file_recom)
    Replay = ReplayMemory(env, policy, capacity, max_length, num_clicks, recom_number, evaluate=True)
    Replay.gen_sample(bsize)
    Replay.write_sample(file_action, file_reward, file_recom, num_clicks, add_end=False)
    orig_reward = Replay.rewards.data.cpu().float().sum(1).mean().numpy()
    print('\nthe original reward is:' + str(orig_reward))


