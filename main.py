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


N_REQ = st.N_REQ
# random_seed = st.random_seed
# if random_seed:
#     torch.manual_seed(random_seed)

USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)

device = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('학습을 진행하는 기기:',device)
# https://github.com/nikhilbarhate99/PPO-PyTorch


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        # actor
        self.action_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.ReLU(),
            # nn.Linear(n_latent_var, n_latent_var),
            # nn.ReLU(),
            nn.Linear(n_latent_var, action_dim),
            nn.Softmax(dim=-1)
        )

        # critic
        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.ReLU(),
            # nn.Linear(n_latent_var, n_latent_var),
            # nn.ReLU(),
            # nn.Linear(n_latent_var, n_latent_var),
            # nn.ReLU(),
            nn.Linear(n_latent_var, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        # print('action_probs', action_probs)


        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # print('update')
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)

        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def save_model(self, path):
        torch.save(self.policy, path)

    def load_model(self, path):
        torch.load(path)


def main():

    n_latent_var = 128  # number of variables in hidden layer
    update_timestep = 100  # update policy every n timesteps
    lr = 0.0002
    betas = (0.9, 0.999)
    gamma = 0.99  # discount factor
    K_epochs = 64  # update policy for K epochs
    eps_clip = 0.1  # clip parameter for PPO

    #############################################
    graph_train = Graph_34()
    graph_test = graph_train
    action_size = len(graph_train.cs_info)
    state_size = action_size * (6 + st.N_SOCKET) + 5

    print(state_size, action_size)

    TRAIN=False
    now_start = datetime.datetime.now()
    resultdir = '{0:02}-{1:02} {2:02}-{3:02} {4:02} {5} {6} {11} {7} reserv Req_{8} Socket_{9} N_node_{10}'.format(
        now_start.month, now_start.day, now_start.hour, now_start.minute, now_start.second, TRAIN, st.EPISODES, 0,
        N_REQ, st.N_SOCKET, graph_train.num_node, st.Bsize)
    basepath = os.getcwd()
    dirpath = os.path.join(basepath, resultdir)
    envr.createFolder(dirpath)



    agent = PPO(state_size, action_size, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    env = Env(graph_train, state_size, action_size)

    memory = Memory()
    print(lr, betas)

    episcores, episodes, eptscores, finalscores = [], [], [], []
    waiting_time_list = []
    driving_time_list = []
    driving_distance_list = []
    timestep = 0
    log_avg_ep = []
    log_avg_rwd = []
    log_tmp_add_ep_rwd = []

    test_list = list(itertools.product(([0, 1, 2]), repeat=10))

    max_reward= -1000000
    max_action_list = []


    for e, kk in enumerate(test_list):

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

        for ac in kk:
            timestep += 1
            action = ac

            action_list.append(action)
            next_state, next_pev, reward, done = env.step(action)
            ept_score += reward * 60
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



        if max_reward<ept_score:
            max_reward = ept_score
            max_action_list = action_list


        print(action_list)
        print(max_reward)
        print(max_action_list)
        episodes.append(e)
        eptscores.append(ept_score)
        finalscores.append(final_score)
        waiting_time_list.append(wt)
        driving_time_list.append(dt)
        driving_distance_list.append(dd)
        log_tmp_add_ep_rwd.append(ept_score)

    # for e in range(st.EPISODES):
    #     episcore = 0
    #     epistep = 0
    #     final_score = 0
    #     ept_score = 0
    #     dt = 0
    #     wt = 0
    #     dd = 0
    #     action_list = []
    #
    #     print("\nEpi:", e, 'episcore:', ept_score)
    #     done = False
    #
    #     state = env.reset()
    #     state = np.reshape(state, [state_size])
    #
    #     while not done:
    #         timestep += 1
    #         action = agent.policy_old.act(state, memory)
    #
    #         action_list.append(action)
    #         next_state, next_pev, reward, done = env.step(action)
    #
    #         next_state = np.reshape(next_state, [state_size])
    #
    #         ept_score += reward*60
    #         state = next_state
    #
    #         if next_pev != -1:
    #             env.pev = next_pev
    #         else:
    #             done = 1
    #
    #         if done:
    #             tot_traveltime = 0
    #             tot_cost = 0
    #             true_waitingtime = 0
    #             true_charging_duration = 0
    #             totaldrivingtime = 0
    #             tot_time_diff = 0.0
    #
    #             for ev in env.request_be_EV:
    #                 tot_traveltime += ev.true_waitingtime + ev.true_charging_duration + ev.totaldrivingtime
    #
    #                 true_waitingtime += ev.true_waitingtime
    #                 true_charging_duration += ev.true_charging_duration
    #                 totaldrivingtime += ev.totaldrivingtime
    #                 tot_cost += ev.totalcost
    #                 dd += ev.totaldrivingdistance
    #                 tot_time_diff += ev.totaldrivingtime - ev.ept_totaldrivingtime
    #                 tot_time_diff += ev.true_waitingtime - ev.ept_waitingtime
    #                 tot_time_diff += ev.true_charging_duration - ev.ept_charging_duration
    #
    #             final_score = tot_traveltime
    #             dt = totaldrivingtime
    #             wt = true_waitingtime
    #             # reward = -tot_time_diff / 60
    #             # ept_score += reward
    #             print('ept_score: ', ept_score)
    #
    #         # Saving reward and is_terminal:
    #         memory.rewards.append(reward*60)
    #         memory.is_terminals.append(done)
    #
    #
    #         # update if its time
    #     if timestep % update_timestep == 0:
    #         agent.update(memory)
    #         memory.clear_memory()
    #         # timestep = 0
    #
    #     print(action_list)
    #     episodes.append(e)
    #     eptscores.append(ept_score)
    #     finalscores.append(final_score)
    #     waiting_time_list.append(wt)
    #     driving_time_list.append(dt)
    #     driving_distance_list.append(dd)
    #
    #     log_tmp_add_ep_rwd.append(ept_score)
    #
    #     if e % 100 == 99:
    #         log_avg_ep.append(e)
    #         avg_rwd = sum(log_tmp_add_ep_rwd)/len(log_tmp_add_ep_rwd)
    #         log_avg_rwd.append(avg_rwd)
    #         log_tmp_add_ep_rwd.clear()
    #
    #
    #     # if e % 100 == 99:
    #     #     agent.save_model("{}/ppo_model_{}.pt".format(resultdir, e))
    #
    #     if e % 100 == 99:
    #         now = datetime.datetime.now()
    #         training_time = now - now_start
    #
    #         plt.title('Training eptscores: {}'.format(training_time))
    #         plt.xlabel('Epoch')
    #         plt.ylabel('score')
    #         plt.plot(episodes, eptscores, 'b')
    #         fig = plt.gcf()
    #         fig.savefig('{}/train eptscores.png'.format(resultdir), facecolor='white', dpi=600)
    #         plt.clf()
    #
    #         plt.title('Training avg eptscores: {}'.format(training_time))
    #         plt.xlabel('Epoch')
    #         plt.ylabel('score')
    #         plt.plot(log_avg_ep, log_avg_rwd, 'b')
    #         fig = plt.gcf()
    #         fig.savefig('{}/train avg eptscores.png'.format(resultdir), facecolor='white', dpi=600)
    #         plt.clf()
    #
    #
    #         plt.title('Training finalscores: {}'.format(training_time))
    #         plt.xlabel('Epoch')
    #         plt.ylabel('step')
    #         plt.plot(episodes, finalscores, 'r')
    #         fig = plt.gcf()
    #         fig.savefig('{}/train finalscores.png'.format(resultdir), facecolor='white', dpi=600)
    #         plt.clf()
    #         ##############################################################################################################
    #         plt.title('Training waiting_time: {}'.format(training_time))
    #         plt.xlabel('Epoch')
    #         plt.ylabel('score')
    #         plt.plot(episodes, waiting_time_list, 'b')
    #         fig = plt.gcf()
    #         fig.savefig('{}/train waiting_time.png'.format(resultdir), facecolor='white', dpi=600)
    #         plt.clf()
    #
    #         plt.title('Training driving_time: {}'.format(training_time))
    #         plt.xlabel('Epoch')
    #         plt.ylabel('step')
    #         plt.plot(episodes, driving_time_list, 'r')
    #         fig = plt.gcf()
    #         fig.savefig('{}/train driving_time.png'.format(resultdir), facecolor='white', dpi=600)
    #         plt.clf()
    #
    #         plt.title('Training distance: {}'.format(training_time))
    #         plt.xlabel('Epoch')
    #         plt.ylabel('step')
    #         plt.plot(episodes, driving_distance_list, 'r')
    #         fig = plt.gcf()
    #         fig.savefig('{}/train driving_distance.png'.format(resultdir), facecolor='white', dpi=600)
    #         plt.clf()

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



#     # TRAIN = True
#     TRAIN = False
#
#     graph_train = Graph_34()
#     graph_test = graph_train
#
#     now_start = datetime.datetime.now()
#     resultdir = '{0:02}-{1:02} {2:02}-{3:02} {4:02} {5} {6} {11} {7} reserv Req_{8} Socket_{9} N_node_{10}'.format(
#         now_start.month, now_start.day, now_start.hour, now_start.minute, now_start.second, TRAIN, setting.EPISODES, setting.EPS_DC,
#         setting.N_REQ, setting.N_SOCKET, graph_train.num_node, setting.Bsize)
#     basepath = os.getcwd()
#     dirpath = os.path.join(basepath, resultdir)
#     ta.createFolder(dirpath)
#
#     action_size = len(graph_train.cs_info)
#     # action_size = N_CS
#     state_size = action_size * (6 + setting.N_SOCKET) + 5
#     # state_size = action_size*(6)+6
#     # state_size = action_size*(5)+3
#
#     print('S:{} A:{}'.format(state_size, action_size))
#
#     # test_result_list
#     for i in range(10000):
#         print(i, 'th test')
#         npev = setting.N_REQ
#         # npev = nEV
#         EV_list, CS_list, graph_test = gen_test_envir_simple(npev, graph_test)
#         # graph = graph_train
#
#         for ev in EV_list:
#             print(ev.id, ev.t_start, ev.curr_SOC)
#
#
# ########################################################################################################################
#         EV_list_short_dist_Greedy = copy.deepcopy(EV_list)
#         CS_list_short_dist_Greedy = copy.deepcopy(CS_list)
#         ta.get_greedy_shortest_fleet(EV_list_short_dist_Greedy, CS_list_short_dist_Greedy, graph_test)
#         tot_wt = 0
#         tot_cost = 0
#         for pev in EV_list_short_dist_Greedy:
#             tot_wt += pev.true_waitingtime
#             tot_cost += pev.totalcost
#         print('==========EV_list_short_dist Greedy===================')
#         print('Avg. total waiting time: ', tot_wt / len(EV_list_short_dist_Greedy))
#         print('Total cost: ', tot_cost)
#

#
# ########################################################################################################################
#
#         EV_list_shorttime_Greedy_reserv = copy.deepcopy(EV_list)
#         CS_list_shorttime_Greedy_reserv = copy.deepcopy(CS_list)
#         ta.get_greedy_shorttime_fleet_reserve(EV_list_shorttime_Greedy_reserv, CS_list_shorttime_Greedy_reserv,
#                                               graph_test)
#
#         tot_wt = 0
#         tot_cost = 0
#         for pev in EV_list_shorttime_Greedy_reserv:
#             tot_wt += pev.true_waitingtime
#             tot_cost += pev.totalcost
#         print('==========EV_list_Greedy===================')
#         print('Avg. total waiting time: ', tot_wt / len(EV_list_shorttime_Greedy_reserv))
#         print('Total cost: ', tot_cost)
#
#
# ########################################################################################################################
#
#         EV_list_shortWT_Greedy_reserv = copy.deepcopy(EV_list)
#         CS_list_shortWT_Greedy_reserv = copy.deepcopy(CS_list)
#         ta.get_greedy_shortwt_fleet_reserv(EV_list_shortWT_Greedy_reserv, CS_list_shortWT_Greedy_reserv, graph_test)
#
#         tot_wt = 0
#         tot_cost = 0
#         for pev in EV_list_shortWT_Greedy_reserv:
#             tot_wt += pev.true_waitingtime
#             tot_cost += pev.totalcost
#         print('==========EV_list_Greedy===================')
#         print('Avg. total waiting time: ', tot_wt / len(EV_list_shortWT_Greedy_reserv))
#         print('Total cost: ', tot_cost)
#
#
#
#
#         ta.sim_result_text_fleet(i, CS_list, graph_test, resultdir,
#                                  MD=(EV_list_short_dist_Greedy, CS_list_short_dist_Greedy),
#                                  MTTnRU=(EV_list_shorttime_Greedy_reserv, CS_list_shorttime_Greedy_reserv),
#                                  MWTnRU=(EV_list_shortWT_Greedy_reserv, CS_list_shortWT_Greedy_reserv))

