import torch
import torch.nn as nn
from torch.distributions import Categorical
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

import collections
import torch.nn.functional as F
import torch.optim as optim


N_REQ = st.N_REQ
# random_seed = st.random_seed
# if random_seed:
#     torch.manual_seed(random_seed)

USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)

device = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('학습을 진행하는 기기:',device)
# https://github.com/nikhilbarhate99/PPO-PyTorch






#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
buffer_limit  = 50000
batch_size    = 32










class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(29, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)
        else:
            return out.argmax().item()


def train(q, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)
        #
        # print(s.shape)
        # print(a.shape)
        # print(r.shape)
        # print(s_prime.shape)

        q_out = q(s)

        # print(q_out.shape)

        q_a = q_out.gather(1, a)
        # print('q_a', q_a.shape)



        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    print_interval=20
  #############################################
    graph_train = Graph_34()
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



    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    env = Env(graph_train, state_size, action_size)


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


        state = env.reset()
        state = np.reshape(state, [state_size])
        # print('state', state.shape)
        epsilon = max(0.01, 0.30 - 0.01 * (e / 200))  # Linear annealing from 8% to 1%
        done = False

        print("\nEpi:", e, 'episcore:', ept_score,  'epsilon:', epsilon)


        while not done:

            action = q.sample_action(torch.from_numpy(state).float(), epsilon)
            next_state, next_pev, reward, done = env.step(action)

            next_state = np.reshape(next_state, [state_size])


            # print('state', state.shape)
            # print('n state',next_state.shape)
            # print('rwd', reward.shape)


            done_mask = 0.0 if done else 1.0

            memory.put((state, action, reward*60, next_state, done_mask))


            timestep += 1

            action_list.append(action)
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
                # reward = -tot_time_diff / 60
                # ept_score += reward
                print('ept_score: ', ept_score)


        if memory.size() > 2000:
            train(q, q_target, memory, optimizer)

        if e % print_interval == 0 and e != 0:
            q_target.load_state_dict(q.state_dict())



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


        # if e % 100 == 99:
        #     agent.save_model("{}/ppo_model_{}.pt".format(resultdir, e))

        if e % 100 == 99:
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



if __name__ == '__main__':


    main()


