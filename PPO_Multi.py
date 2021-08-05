import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.multiprocessing as mp
import gym
import time
import matplotlib.pyplot as plt
from Graph import Graph_34
from Env import Env


# USE_CUDA = torch.cuda.is_available()
# print(USE_CUDA)
#
# device = torch.device('cuda:0' if USE_CUDA else 'cpu')
device = torch.device('cpu')
# print('학습을 진행하는 기기000000:',device)
# https://github.com/nikhilbarhate99/PPO-PyTorch


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 800),
            nn.ReLU(),
            nn.Linear(800, 800),
            nn.ReLU(),
            nn.Linear(800, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 800),
            nn.ReLU(),
            nn.Linear(800, 800),
            nn.ReLU(),
            nn.Linear(800, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )


    def forward(self):
        raise NotImplementedError

    def act(self, state):

        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):

        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy

    def save(self, checkpoint_path):
        torch.save(self.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


class PPO(mp.Process):
    def __init__(self, id, update_timestep, gmodel, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, queue):
        super(PPO, self).__init__()

        self.id = id

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.queue = queue
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.policy.load_state_dict(gmodel.state_dict())
        self.gmodel = gmodel

        self.N_A = action_dim
        self.N_S = state_dim

        # print('In class')
        # for param_tensor in self.gmodel.state_dict():
        #     print(param_tensor, "\t", self.gmodel.state_dict()[param_tensor])

        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.MseLoss = nn.MSELoss()

        # self.env = gym.make('CartPole-v0').unwrapped
        self.num_EP = 500000
        self.update_timestep =  update_timestep

        print(self.id, 'is On')

    def run(self):
        print('run ')
        timestep = 0
        cnt = 0

        graph = Graph_34()
        env = Env(graph, self.N_S, self.N_A, 0)

        episcores, episodes, eptscores, finalscores = [], [], [], []
        waiting_time_list = []
        driving_time_list = []
        driving_distance_list = []
        timestep = 0
        log_avg_ep = []
        log_avg_rwd = []
        log_tmp_add_ep_rwd = []

        for e in range(self.num_EP):
            episcore = 0
            epistep = 0
            final_score = 0
            ept_score = 0
            dt = 0
            wt = 0
            dd = 0

            done = False
            state = env.reset()
            while not done:
                timestep += 1

                action = self.select_action(state)

                next_state, next_pev, reward, done = env.step(action)
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

                    final_score = tot_traveltime / 60
                    dt = totaldrivingtime
                    wt = true_waitingtime
                    # reward += -tot_time_diff
                    # ept_score += -tot_time_diff

                    self.queue.put(ept_score)

                self.buffer.rewards.append(reward*60)
                self.buffer.is_terminals.append(done)

                # print(timestep, done)
                if timestep % self.update_timestep == 0:
                    # print(self.id, 'len',len(self.buffer.states))

                    self.update()
                    self.policy.load_state_dict(self.gmodel.state_dict())

        self.queue.put(None)






    def select_action(self, state):

        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action.item()

    def update(self):

        # Monte Carlo estimate of returns
        # print('Update')
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + self.MseLoss(state_values, rewards.detach())
            # loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            # loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(state_values, rewards.detach())

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.gmodel.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()


    def save(self, checkpoint_path):
        torch.save(self.policy.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))