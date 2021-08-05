import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym, os
import Env as envr
# from Env import Env
# from Env import init_request
# import CS
# from  CS import reset_CS_info
import setting as st
# import matplotlib.pyplot as plt
from Graph import Graph_34
import datetime
# import random
# import itertools
# import numpy as np
# from torch.distributions import MultivariateNormal
# import torch.nn.functional as F
# import MainLogic as ta
# import copy
# from Env import init_request
# from  CS import reset_CS_info
import torch.multiprocessing as mp
# from multiprocessing import Lock, Queue, current_process
import time, os
from PPO_Multi import PPO, ActorCritic
import matplotlib.pyplot as plt




if __name__ == '__main__':
    print('what the')
    gamma = 0.99  # discount factor
    K_epochs = 64  # update policy for K epochs
    n_pro = 16
    update_timestep = st.N_REQ * 1000
    eps_clip = 0.1  # clip parameter for PPO
    lr_actor = 0.0001  # learning rate for actor network
    lr_critic = 0.0001  # learning rate for critic network
    # env_name = "CartPole-v1"
    # env = gym.make(env_name)
    # N_S = env.observation_space.shape[0]
    # N_A = env.action_space.n


    graph = Graph_34()
    N_A = len(graph.cs_info)
    N_S = N_A * (6 + st.N_SOCKET) + 3 + 68

    gmodel = ActorCritic(N_S, N_A)
    gmodel.share_memory()

    now_start = datetime.datetime.now()
    resultdir = '{0:02}-{1:02} {2:02}-{3:02} {4:02}'.format(now_start.month, now_start.day, now_start.hour, now_start.minute, now_start.second)
    basepath = os.getcwd()
    dirpath = os.path.join(basepath, resultdir)
    envr.createFolder(dirpath)

    pthpath = os.path.join(dirpath, 'model')
    envr.createFolder(pthpath)


    start_time = time.time()
    num = mp.Value('i', 0)
    res_queue = mp.Queue()

    agents = [PPO(i, update_timestep, gmodel, N_S, N_A, lr_actor, lr_critic, gamma, K_epochs, eps_clip, res_queue) for i in range(n_pro)]
    [ag.start() for ag in agents]

    res = []  # record episode reward to plot
    log_avg_ep = []
    log_avg_rwd = []
    log_tmp_add_ep_rwd = []

    flag=0
    cnt=0
    avg_rwd=0
    while True:
        r = res_queue.get()
        cnt+=1
        if r is not None:
            res.append(r)

            log_tmp_add_ep_rwd.append(r)
            if cnt % 2000 == 0:
                log_avg_ep.append(cnt)
                avg_rwd = sum(log_tmp_add_ep_rwd) / len(log_tmp_add_ep_rwd)
                log_avg_rwd.append(avg_rwd)
                log_tmp_add_ep_rwd.clear()
                print(cnt, 'Avg. Reward: ', avg_rwd)

                training_time = datetime.datetime.now() - now_start

                plt.title('Training avg eptscores: {}'.format(training_time))
                plt.xlabel('Epoch')
                plt.ylabel('score')
                plt.plot(log_avg_ep, log_avg_rwd, 'b')
                plt.grid(True, axis='y')
                fig = plt.gcf()
                fig.savefig('{}/train avg eptscores.png'.format(resultdir), facecolor='white', dpi=600)
                plt.clf()
            if cnt % 1000 == 0:
                file_name = "PPO_{:05d}_{}.pth".format(cnt, round(avg_rwd))
                file_path = os.path.join(pthpath, file_name)
                gmodel.save(file_path)

        else:
            flag+=1
            if flag==n_pro:
                break

    [ag.join() for ag in agents]

    plt.plot(res)
    plt.show()

    elapsed_time = time.time() - start_time

    print(num.value)
    print(elapsed_time)






