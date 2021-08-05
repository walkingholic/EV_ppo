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
from Env import init_request
from  CS import reset_CS_info

# from PPO_LSTM import PPO



N_REQ = st.N_REQ


def main(graph):

    # n_latent_var = 128  # number of variables in hidden layer
    update_timestep = 200  # update policy every n timesteps

    gamma = 0.99  # discount factor
    K_epochs = 4  # update policy for K epochs
    eps_clip = 0.1  # clip parameter for PPO
    lr_actor = 0.0001  # learning rate for actor network
    lr_critic = 0.0001  # learning rate for critic network

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
    envr.createFolder(dirpath)

    pthpath = os.path.join(dirpath, 'model')
    envr.createFolder(pthpath)


    agent = PPO(state_size, action_size, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
    random_seed = 0
    # env = Env(graph_train, state_size, action_size, random_seed, request_EV, CS_origin_list)
    env = Env(graph_train, state_size, action_size, random_seed)

    # EV_list = init_request(N_REQ, graph_train, 0)
    # CS_list = reset_CS_info(graph_train)



    episcores, episodes, eptscores, finalscores = [], [], [], []
    waiting_time_list = []
    driving_time_list = []
    driving_distance_list = []
    timestep = 0
    log_avg_ep = []
    log_avg_rwd = []
    log_tmp_add_ep_rwd = []

    for e in range(st.EPISODES):
        episcore = 0
        epistep = 0
        final_score = 0
        ept_score = 0
        dt = 0
        wt = 0
        dd = 0
        action_list = []

        print("\nEpi:", e, 'episcore:', ept_score)
        done = False

        state = env.reset()
        # state = env.test_reset(graph_train, copy.deepcopy(EV_list), copy.deepcopy(CS_list))

        # state = np.reshape(state, [state_size])

        while not done:
            timestep += 1
            action = agent.select_action(state)

            action_list.append(action)
            next_state, next_pev, reward, done = env.step(action)

            # next_state = np.reshape(next_state, [state_size])

            ept_score += reward
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
                    # if e>4000:
                    #     print(ev.id, ev.true_waitingtime, ev.true_charging_duration, ev.totaldrivingtime)
                    tot_traveltime += ev.true_waitingtime + ev.true_charging_duration + ev.totaldrivingtime

                    true_waitingtime += ev.true_waitingtime
                    true_charging_duration += ev.true_charging_duration
                    totaldrivingtime += ev.totaldrivingtime
                    tot_cost += ev.totalcost
                    dd += ev.totaldrivingdistance
                    tot_time_diff += ev.totaldrivingtime - ev.ept_totaldrivingtime
                    tot_time_diff += ev.true_waitingtime - ev.ept_waitingtime
                    tot_time_diff += ev.true_charging_duration - ev.ept_charging_duration

                final_score = tot_traveltime/60
                dt = totaldrivingtime
                wt = true_waitingtime
                # reward += -tot_time_diff
                # ept_score += -tot_time_diff



                print('ept_score: ', ept_score)
                print('tot_traveltime: ', tot_traveltime)
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)

            # print(timestep, state, next_state, reward, done)





            # update if its time
        # if timestep % update_timestep == 0:
        if done:
            agent.update()


        print(action_list)
        episodes.append(e)
        eptscores.append(ept_score)
        finalscores.append(final_score)
        waiting_time_list.append(wt)
        driving_time_list.append(dt)
        driving_distance_list.append(dd)

        log_tmp_add_ep_rwd.append(ept_score)

        if e % 100 == 99:
            log_avg_ep.append(e)
            avg_rwd = sum(log_tmp_add_ep_rwd)/len(log_tmp_add_ep_rwd)
            log_avg_rwd.append(avg_rwd)
            log_tmp_add_ep_rwd.clear()


        if e >50000 and e% 1000 == 0:
            file_name = "PPO_{:05d}_{}.pth".format(e, round(avg_rwd))
            file_path = os.path.join(pthpath, file_name)
            agent.save(file_path)
            # timestep = 0


        if e % 1000 == 999:
            now = datetime.datetime.now()
            training_time = now - now_start


            plt.title('Training avg eptscores: {}'.format(training_time))
            plt.xlabel('Epoch')
            plt.ylabel('score')
            plt.plot(log_avg_ep, log_avg_rwd, 'b')
            plt.grid(True, axis='y')
            fig = plt.gcf()
            fig.savefig('{}/train avg eptscores.png'.format(resultdir), facecolor='white', dpi=600)
            plt.clf()

            # plt.title('Training eptscores: {}'.format(training_time))
            # plt.xlabel('Epoch')
            # plt.ylabel('score')
            # plt.plot(episodes, eptscores, 'b')
            # plt.grid(True, axis='y')
            # fig = plt.gcf()
            # fig.savefig('{}/train eptscores.png'.format(resultdir), facecolor='white', dpi=600)
            # plt.clf()
            #
            # plt.title('Training finalscores: {}'.format(training_time))
            # plt.xlabel('Epoch')
            # plt.ylabel('step')
            # plt.plot(episodes, finalscores, 'r')
            # plt.grid(True, axis='y')
            # fig = plt.gcf()
            # fig.savefig('{}/train finalscores.png'.format(resultdir), facecolor='white', dpi=600)
            # plt.clf()
            # ##############################################################################################################
            # plt.title('Training waiting_time: {}'.format(training_time))
            # plt.xlabel('Epoch')
            # plt.ylabel('score')
            # plt.plot(episodes, waiting_time_list, 'b')
            # plt.grid(True, axis='y')
            # fig = plt.gcf()
            # fig.savefig('{}/train waiting_time.png'.format(resultdir), facecolor='white', dpi=600)
            # plt.clf()
            #
            # plt.title('Training driving_time: {}'.format(training_time))
            # plt.xlabel('Epoch')
            # plt.ylabel('step')
            # plt.plot(episodes, driving_time_list, 'r')
            # plt.grid(True, axis='y')
            # fig = plt.gcf()
            # fig.savefig('{}/train driving_time.png'.format(resultdir), facecolor='white', dpi=600)
            # plt.clf()
            #
            # plt.title('Training distance: {}'.format(training_time))
            # plt.xlabel('Epoch')
            # plt.ylabel('step')
            # plt.plot(episodes, driving_distance_list, 'r')
            # plt.grid(True, axis='y')
            # fig = plt.gcf()
            # fig.savefig('{}/train driving_distance.png'.format(resultdir), facecolor='white', dpi=600)
            # plt.clf()

    now = datetime.datetime.now()
    training_time = now - now_start

    fw = open('{}/epi_score.txt'.format(resultdir), 'w', encoding='UTF8')
    for sc in eptscores:
        fw.write(str(sc) + '\t')
    fw.write('\n')
    for fsc in finalscores:
        fw.write(str(fsc) + '\t')

    fw.close()



if __name__ == '__main__':

    seed = 0
    graph = Graph_34()

    main(graph)

    # st = list(np.eye(5)[2])
    #
    # st = st+list(np.eye(5)[1])
    # print(st)
    # tt = [1,23,2,1,2,3]
    # t2 = [11,22,33,44]
    # tt += t2
    # print(tt)








