
import gym, os
import Env as envr
from Env import Env
import CS
import setting as st
import matplotlib.pyplot as plt
from Graph import Graph_34
import datetime
import random
import itertools
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import copy
from Env import init_request
from  CS import reset_CS_info


N_REQ = st.N_REQ
# random_seed = st.random_seed
# if random_seed:
#     torch.manual_seed(random_seed)

USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)

device = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('학습을 진행하는 기기:',device)
# https://github.com/nikhilbarhate99/PPO-PyTorch


# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
lmbda = 0.95
eps_clip = 0.1
K_epoch = 32
update_timestep = 100


class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(29, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_pi = nn.Linear(256, 3)
        self.fc_v = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s, a, r, s_prime, done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                              torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
                                              torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1, a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s), td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()


def main():

    # n_latent_var = 128  # number of variables in hidden layer
    # update_timestep = 100  # update policy every n timesteps
    # lr = 0.0002
    # betas = (0.9, 0.999)
    # gamma = 0.99  # discount factor
    # K_epochs = 64  # update policy for K epochs
    # eps_clip = 0.1  # clip parameter for PPO

    #############################################
    graph_train = Graph_34()
    graph_test = graph_train
    action_size = len(graph_train.cs_info)
    state_size = action_size * (6 + st.N_SOCKET) + 5

    print(state_size, action_size)

    TRAIN = False
    now_start = datetime.datetime.now()
    resultdir = '{0:02}-{1:02} {2:02}-{3:02} {4:02} {5} {6} {11} {7} reserv Req_{8} Socket_{9} N_node_{10}'.format(
        now_start.month, now_start.day, now_start.hour, now_start.minute, now_start.second, TRAIN, st.EPISODES, 0,
        N_REQ, st.N_SOCKET, graph_train.num_node, st.Bsize)
    basepath = os.getcwd()
    dirpath = os.path.join(basepath, resultdir)
    envr.createFolder(dirpath)

    model = PPO()

    env = Env(graph_train, state_size, action_size, 0)
    EV_list = init_request(N_REQ, graph_train, 0)
    CS_list = reset_CS_info(graph_train)

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
        state = env.test_reset(graph_train, copy.deepcopy(EV_list), copy.deepcopy(CS_list))
        state = np.reshape(state, [state_size])

        while not done:
            timestep += 1
            prob =  model.pi(torch.from_numpy(state).float())
            m = Categorical(prob)
            action = m.sample().item()
            action_list.append(action)
            next_state, next_pev, reward, done = env.step(action)
            next_state = np.reshape(next_state, [state_size])
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
                # reward = -tot_time_diff / 60
                # ept_score += reward
                print('ept_score: ', ept_score)

            model.put_data((state, action, reward*60, next_state, prob[action].item(), done))
            state = next_state

            if timestep % update_timestep == 0:
                model.train_net()
        # model.train_net()


        print(action_list)
        episodes.append(e)
        eptscores.append(ept_score)
        finalscores.append(final_score)
        waiting_time_list.append(wt)
        driving_time_list.append(dt)
        driving_distance_list.append(dd)

        log_tmp_add_ep_rwd.append(ept_score)

        if e % 500 == 499:
            log_avg_ep.append(e)
            avg_rwd = sum(log_tmp_add_ep_rwd)/len(log_tmp_add_ep_rwd)
            log_avg_rwd.append(avg_rwd)
            log_tmp_add_ep_rwd.clear()


        # if e % 100 == 99:
        #     agent.save_model("{}/ppo_model_{}.pt".format(resultdir, e))

        if e % 500 == 499:
            now = datetime.datetime.now()
            training_time = now - now_start


            plt.title('Training avg eptscores: {}'.format(training_time))
            plt.xlabel('Epoch')
            plt.ylabel('score')
            plt.plot(log_avg_ep, log_avg_rwd, 'b')
            fig = plt.gcf()
            fig.savefig('{}/train avg eptscores.png'.format(resultdir), facecolor='white', dpi=600)
            plt.clf()

            plt.title('Training eptscores: {}'.format(training_time))
            plt.xlabel('Epoch')
            plt.ylabel('score')
            plt.plot(episodes, eptscores, 'b')
            fig = plt.gcf()
            fig.savefig('{}/train eptscores.png'.format(resultdir), facecolor='white', dpi=600)
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



if __name__ == '__main__':


    main()
