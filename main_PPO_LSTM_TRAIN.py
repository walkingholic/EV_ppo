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
# from PPO import PPO
from PPO_LSTM import PPO



N_REQ = st.N_REQ


def main(graph):

    # n_latent_var = 128  # number of variables in hidden layer
    update_timestep = 100  # update policy every n timesteps

    gamma = 0.99  # discount factor
    K_epochs = 64  # update policy for K epochs
    eps_clip = 0.1  # clip parameter for PPO
    lr_actor = 0.0002  # learning rate for actor network
    lr_critic = 0.0005  # learning rate for critic network


    #############################################
    graph_train = graph
    action_size = len(graph_train.cs_info)
    state_size = action_size * (6 + st.N_SOCKET) + 5

    # print(state_size, action_size)

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


    agent = PPO(state_size, action_size, lr_actor, gamma, K_epochs, eps_clip)
    random_seed = 0

    # env = Env(graph_train, state_size, action_size, random_seed, request_EV, CS_origin_list)
    env = Env(graph_train, state_size, action_size, random_seed)


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
        state = np.reshape(state, [1,state_size])
        h_out = (torch.zeros([1, 1, 128], dtype=torch.float), torch.zeros([1, 1, 128], dtype=torch.float))

        while not done:
            timestep += 1

            # print('state', state.shape)
            h_in = h_out
            prob, h_out = agent.pi(torch.from_numpy(state).float(), h_in)
            prob = prob.view(-1)
            m = Categorical(prob)
            action = m.sample().item()


            # print(h_out)
            # print(h_out[0].shape)
            # print(h_out[1].shape)

            action_list.append(action)
            next_state, next_pev, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1,state_size])
            # print('next_state', next_state.shape)

            ept_score += reward*60

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
                # reward += -tot_time_diff
                # ept_score += -tot_time_diff



                print('ept_score: ', ept_score)
                print('tot_traveltime: ', tot_traveltime)
            agent.put_data((state, action, reward*60, next_state, prob[action].item(), h_in, h_out, done))
            state = next_state
            # print(timestep, state, next_state, reward, done)

            # update if its time
        if timestep % update_timestep == 0:
            agent.train_net()


        print(action_list)
        episodes.append(e)
        eptscores.append(ept_score)
        finalscores.append(final_score)
        waiting_time_list.append(wt)
        driving_time_list.append(dt)
        driving_distance_list.append(dd)

        log_tmp_add_ep_rwd.append(ept_score)

        if e % 200 == 99:
            log_avg_ep.append(e)
            avg_rwd = sum(log_tmp_add_ep_rwd)/len(log_tmp_add_ep_rwd)
            log_avg_rwd.append(avg_rwd)
            log_tmp_add_ep_rwd.clear()


        if e >50 and e% 200 == 0:
            file_name = "PPO_{:05d}_{}.pth".format(e, round(avg_rwd))
            file_path = os.path.join(pthpath, file_name)
            agent.save(file_path)
            # timestep = 0


        if e % 500 == 99:
            now = datetime.datetime.now()
            training_time = now - now_start

            plt.title('Training eptscores: {}'.format(training_time))
            plt.xlabel('Epoch')
            plt.ylabel('score')
            plt.plot(episodes, eptscores, 'b')
            fig = plt.gcf()
            fig.savefig('{}/train eptscores.png'.format(resultdir), facecolor='white', dpi=600)
            plt.clf()

            plt.title('Training avg eptscores: {}'.format(training_time))
            plt.xlabel('Epoch')
            plt.ylabel('score')
            plt.plot(log_avg_ep, log_avg_rwd, 'b')
            fig = plt.gcf()
            fig.savefig('{}/train avg eptscores.png'.format(resultdir), facecolor='white', dpi=600)
            plt.clf()


            plt.title('Training finalscores: {}'.format(training_time))
            plt.xlabel('Epoch')
            plt.ylabel('step')
            plt.plot(episodes, finalscores, 'r')
            fig = plt.gcf()
            fig.savefig('{}/train finalscores.png'.format(resultdir), facecolor='white', dpi=600)
            plt.clf()
            ##############################################################################################################
            plt.title('Training waiting_time: {}'.format(training_time))
            plt.xlabel('Epoch')
            plt.ylabel('score')
            plt.plot(episodes, waiting_time_list, 'b')
            fig = plt.gcf()
            fig.savefig('{}/train waiting_time.png'.format(resultdir), facecolor='white', dpi=600)
            plt.clf()

            plt.title('Training driving_time: {}'.format(training_time))
            plt.xlabel('Epoch')
            plt.ylabel('step')
            plt.plot(episodes, driving_time_list, 'r')
            fig = plt.gcf()
            fig.savefig('{}/train driving_time.png'.format(resultdir), facecolor='white', dpi=600)
            plt.clf()

            plt.title('Training distance: {}'.format(training_time))
            plt.xlabel('Epoch')
            plt.ylabel('step')
            plt.plot(episodes, driving_distance_list, 'r')
            fig = plt.gcf()
            fig.savefig('{}/train driving_distance.png'.format(resultdir), facecolor='white', dpi=600)
            plt.clf()

    now = datetime.datetime.now()
    training_time = now - now_start

    fw = open('{}/epi_score.txt'.format(resultdir), 'w', encoding='UTF8')
    for sc in eptscores:
        fw.write(str(sc) + '\t')
    fw.write('\n')
    for fsc in finalscores:
        fw.write(str(fsc) + '\t')

    fw.close()


def main_cart():
    learning_rate = 0.0005
    gamma = 0.98
    lmbda = 0.95
    eps_clip = 0.1
    K_epoch = 2
    T_horizon = 20

    env = gym.make('CartPole-v1')
    model = PPO(4, 2, learning_rate, gamma, K_epoch, eps_clip)
    score = 0.0
    print_interval = 20

    for n_epi in range(10000):
        h_out = (torch.zeros([1, 1, 32], dtype=torch.float), torch.zeros([1, 1, 32], dtype=torch.float))
        s = env.reset()
        done = False

        while not done:
            for t in range(T_horizon):
                h_in = h_out
                prob, h_out = model.pi(torch.from_numpy(s).float(), h_in)
                prob = prob.view(-1)
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, info = env.step(a)

                model.put_data((s, a, r / 100.0, s_prime, prob[a].item(), h_in, h_out, done))
                s = s_prime

                score += r
                if done:
                    break

            model.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score / print_interval))
            score = 0.0

    env.close()

if __name__ == '__main__':

    seed = 0
    graph = Graph_34()

    # main(graph)
    main_cart()





