import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym, os
import Env as envr
from Env import Env
from Env import init_request
import CS
from  CS import reset_CS_info
import setting as st
import matplotlib.pyplot as plt
from Graph import Graph_34
import datetime
import random
import itertools
import numpy as np
from torch.distributions import MultivariateNormal
import torch.nn.functional as F
import MainLogic as ta
import copy
from PPO import PPO



N_REQ = st.N_REQ
# random_seed = st.random_seed
# if random_seed:
#     torch.manual_seed(random_seed)

def Test(graph, request_EV, CS_origin_list, model_src):

    # n_latent_var = 128  # number of variables in hidden layer
    update_timestep = 100  # update policy every n timesteps

    gamma = 0.99  # discount factor
    K_epochs = 64  # update policy for K epochs
    eps_clip = 0.1  # clip parameter for PPO
    lr_actor = 0.0001  # learning rate for actor network
    lr_critic = 0.001  # learning rate for critic network

    has_continuous_action_space=False
    action_std = None

    #############################################
    graph_train = graph
    graph_test = graph_train
    action_size = len(graph_train.cs_info)
    state_size = action_size * (6 + st.N_SOCKET) + 3 +68

    print(state_size, action_size)

    TRAIN=False
    now_start = datetime.datetime.now()
    resultdir = '{0:02}-{1:02} {2:02}-{3:02} {4:02} {5} {6} {11} {7} reserv Req_{8} Socket_{9} N_node_{10}'.format(
        now_start.month, now_start.day, now_start.hour, now_start.minute, now_start.second, TRAIN, st.EPISODES, 0,
        N_REQ, st.N_SOCKET, graph_train.num_node, st.Bsize)
    basepath = os.getcwd()
    dirpath = os.path.join(basepath, resultdir)
    # envr.createFolder(dirpath)

    agent = PPO(state_size, action_size, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
    random_seed = 0

    # model_path = os.path.join(basepath, 'test.pth')
    agent.load(model_src)

    env = Env(graph_train, state_size, action_size, random_seed)


    episcores, episodes, eptscores, finalscores = [], [], [], []
    waiting_time_list = []
    driving_time_list = []
    driving_distance_list = []
    timestep = 0


    episcore = 0
    epistep = 0
    final_score = 0
    ept_score = 0
    dt = 0
    wt = 0
    dd = 0
    action_list = []

    state = env.test_reset(graph_train, request_EV, CS_origin_list)
    state = np.reshape(state, [state_size])
    done = False

    while not done:
        timestep += 1
        action = agent.select_action(state)

        action_list.append(action)
        next_state, next_pev, reward, done = env.step(action)

        next_state = np.reshape(next_state, [state_size])

        ept_score += reward*60
        state = next_state

        if next_pev != -1:
            env.pev = next_pev
        else:
            done = 1

        if done:
            tot_traveltime = 0
            tot_cost = 0
            true_waitingtime = 0
            true_charging_duration = 0
            totaldrivingtime = 0
            tot_time_diff = 0.0

            for ev in env.request_be_EV:

                tot_traveltime += ev.true_waitingtime + ev.true_charging_duration + ev.totaldrivingtime

                true_waitingtime += ev.true_waitingtime
                true_charging_duration += ev.true_charging_duration
                totaldrivingtime += ev.totaldrivingtime
                tot_cost += ev.totalcost
                dd += ev.totaldrivingdistance
                tot_time_diff += ev.totaldrivingtime - ev.ept_totaldrivingtime
                tot_time_diff += ev.true_waitingtime - ev.ept_waitingtime
                tot_time_diff += ev.true_charging_duration - ev.ept_charging_duration

            final_score = tot_traveltime
            dt = totaldrivingtime
            wt = true_waitingtime
            reward += -tot_time_diff
            ept_score += -tot_time_diff

            print('ept_score: ', ept_score)
            print('tot_traveltime: ', tot_traveltime)



    if timestep % update_timestep == 0:
        agent.update()


    print(action_list)
    eptscores.append(ept_score)
    finalscores.append(final_score)
    waiting_time_list.append(wt)
    driving_time_list.append(dt)
    driving_distance_list.append(dd)

    return tot_traveltime


if __name__ == '__main__':

    data = []
    for t in range(5):

        tdata = []
        seed = 0
        graph = Graph_34()

        EV_list = init_request(N_REQ, graph, seed)
        CS_list = reset_CS_info(graph)

        EV_list_shorttime_Greedy_reserv = copy.deepcopy(EV_list)
        CS_list_shorttime_Greedy_reserv = copy.deepcopy(CS_list)
        ta.get_greedy_shorttime_fleet_reserve(EV_list_shorttime_Greedy_reserv, CS_list_shorttime_Greedy_reserv,
                                              graph)
        tot_wt = 0
        tot_cost = 0
        tot_traveltime=0
        ev_cs = []
        for pev in EV_list_shorttime_Greedy_reserv:
            ev_cs.append(pev.cs.id)
            tot_traveltime += pev.true_waitingtime + pev.true_charging_duration + pev.totaldrivingtime
            # print(pev.id, pev.true_waitingtime, pev.true_charging_duration, pev.totaldrivingtime, pev.ept_totaldrivingtime)
            tot_wt += pev.true_waitingtime
            tot_cost += pev.totalcost
        print('==========EV_list_Greedy===================')
        print('tot_traveltime: ', tot_traveltime)
        # print('Total cost: ', tot_cost)
        print(ev_cs)
        tdata.append(tot_traveltime)

        basePath = os.getcwd()
        testPath = os.path.join(basePath, 'testModel')
        # print(testPath)
        fileList = os.listdir(testPath)
        for file in fileList:
            if file.endswith('pth'):
                print(file)
                src = os.path.join(testPath, file)
                EV_list_PPO = copy.deepcopy(EV_list)
                CS_list_PPO = copy.deepcopy(CS_list)
                ttt = Test(graph, EV_list_PPO, CS_list_PPO, src)

                tdata.append(ttt)
        data.append(tdata)

    # print(data)
    nd = np.array(data)
    print(nd.T)
    plt.plot(nd.T)
    plt.show()
    plt.plot(nd)
    plt.show()
