import setting
import numpy as np
from EV import EV
import CS
import random
import MainLogic as ta
import os
import setting as st
import copy

# seed = st.random_seed
# np.random.seed(seed)
# random.seed(seed)

class Env:
    def __init__(self, graph,  state_size, action_size, seed, request_EV=None, CS_origin_list=None):

        self.graph = graph
        self.source_node_list = list(self.graph.source_node_set)
        self.destination_node_list = list(self.graph.destination_node_set)
        self.num_request = setting.N_REQ
        self.request_origin_EV = []
        self.request_be_EV = []
        self.request_ing_EV = []
        self.request_ed_EV = []

        self.path_info = []
        self.sim_time=0
        self.CS_origin_list = []
        self.CS_list = []
        self.pev = None

        self.seed = seed

        # self.target = -1
        self.state_size = state_size
        self.action_size = action_size

        self.graph.reset_traffic_info()
        # self.CS_list = CS.reset_CS_info(self.graph)

        if request_EV is None:
            self.request_origin_EV = []
        # else:
        #     self.request_origin_EV = copy.deepcopy(request_EV)

        if CS_origin_list is None:
            self.CS_origin_list = []
        # else:
        #     self.CS_origin_list = copy.deepcopy(CS_origin_list)

        # self.reset()


    def reset(self):
        if self.seed:
            np.random.seed(self.seed)
            random.seed(self.seed)

        self.request_be_EV = init_request(self.num_request, self.graph, self.seed)
        self.CS_list = CS.reset_CS_info(self.graph)

        # for ev in self.request_be_EV:
        #     ev.print()


        self.pev = self.request_be_EV[0]
        self.graph.reset_traffic_info()


        # for ev in self.request_be_EV:
        #     print(ev.id, ev.source, ev.destination, ev.init_SOC)


        self.path_info = []
        self.sim_time = self.pev.t_start
        self.timeIDX = int(self.sim_time / 5)

        self.path_info = ta.get_feature_state_DQN_time_fleet(self.sim_time, self.pev, self.CS_list, self.graph, len(self.CS_list))

        state = list(np.eye(34)[self.pev.curr_location - 1])
        state += list(np.eye(34)[self.pev.destination - 1])
        state += [self.pev.curr_SOC ,(self.pev.t_start/60)/288, 0]
        # state = [self.pev.curr_SOC]

        for path in self.path_info:
            (cs, weight, chargingprice, ept_driving_cost, front_path, rear_path, front_path_distance, rear_path_distance,
             ept_front_d_time,
             ept_rear_d_time, fpath_weight, rpath_weight, ept_WT, ept_charduration, ept_cs_charging_cost,
             ept_home_charging_cost, ept_arrtime, tmp_avail) = path

            for av in tmp_avail:
                cpstate = 0
                if av < self.pev.curr_time:
                    cpstate = 0
                else:
                    cpstate = (av - self.pev.curr_time) / 60
                state += [cpstate]


            state += [ept_WT/60, ept_charduration/60, ept_front_d_time/60 , ept_rear_d_time/60, front_path_distance/20, rear_path_distance/20] # , (ept_arrtime/60)/288

        # state = np.reshape(state, [1, self.state_size])

        return state

    def test_reset(self,graph, EV_list, CS_list):
        # np.random.seed(seed)
        # random.seed(seed)
        self.CS_list = CS_list
        self.request_be_EV = EV_list
        self.graph = graph

        self.pev = self.request_be_EV[0]
        self.path_info = []
        self.sim_time = self.pev.t_start
        self.timeIDX = int(self.sim_time / 5)

        # self.path_info = ta.get_feature_state_DQN_fleet(self.sim_time, self.pev, self.CS_list, self.graph, NCS)
        self.path_info = ta.get_feature_state_DQN_time_fleet(self.sim_time, self.pev, self.CS_list, self.graph, len(self.CS_list))

        # state = [self.pev.curr_location / self.graph.num_node, self.pev.destination / self.graph.num_node, self.pev.curr_SOC, self.pev.req_SOC, (self.pev.t_start/60)/288, 0]
        state = list(np.eye(34)[self.pev.curr_location-1])
        state += list(np.eye(34)[self.pev.destination - 1])
        state += [self.pev.curr_SOC, (self.pev.t_start/60)/288, 0]
        # state = [self.pev.curr_SOC]

        for path in self.path_info:
            (cs, weight, chargingprice, ept_driving_cost, front_path, rear_path, front_path_distance, rear_path_distance,
             ept_front_d_time,
             ept_rear_d_time, fpath_weight, rpath_weight, ept_WT, ept_charduration, ept_cs_charging_cost,
             ept_home_charging_cost, ept_arrtime, tmp_avail) = path

            # cs.update_avail_time(self.pev.curr_time, self.graph)
            # tmp_avail = cs.get_ept_avail_time(ept_arrtime)

            for av in tmp_avail:
                cpstate = 0
                if av < self.pev.curr_time:
                    cpstate = 0
                else:
                    cpstate = (av - self.pev.curr_time) / 60
                state += [cpstate]


            # for cp in cs.cplist:
            #     cpstate = 0
            #     if cp.avail_time < self.pev.curr_time:
            #         cpstate = 0
            #     else:
            #         cpstate = (cp.avail_time - self.pev.curr_time) / 60
            #     state += [cpstate]
            state += [ept_WT/60, ept_charduration/60, ept_front_d_time/60 , ept_rear_d_time/60, front_path_distance/20, rear_path_distance/20] # , (ept_arrtime/60)/288
            # state += [ept_WT/60, ept_charduration/60, ept_front_d_time/60 , ept_rear_d_time/60]
            # state += [ept_WT / 60]
            # state += [self.pev.ept_totalcost/100, ept_WT/60]
        # state = np.reshape(state, [1, self.state_size])

        return state


    def step(self, action):


        (cs, weight, chargingprice, ept_driving_cost, front_path, rear_path, front_path_distance, rear_path_distance, ept_front_d_time,
         ept_rear_d_time, fpath_weight, rpath_weight, ept_WT, ept_charduration, ept_cs_charging_cost,
         ept_home_charging_cost, ept_arrtime, tmp_avail) = self.path_info[action]

        # if cs == None:
        #     reward, done = -50, 1
        #     return np.zeros((1, self.state_size)), -1, reward, done
        #
        # else:

        pev = self.pev
        # pev.print()

        evcango = pev.init_SOC * pev.maxBCAPA / pev.ECRate

        if evcango < front_path_distance:
            reward = -500
            done = 1
            # print(reward)
            return np.zeros((1, self.state_size)), -1, reward, done
        else:
            pev.front_path = front_path
            pev.rear_path = rear_path
            # pev.fdist = front_path_distance
            # pev.rdist = rear_path_distance
            pev.path = front_path + rear_path[1:]
            pev.ept_arrtime = ept_arrtime

            pev.true_arrtime = ta.get_true_arrtime(pev, cs, self.graph)

            pev.ept_waitingtime = ept_WT
            pev.ept_totaldrivingtime = ept_front_d_time + ept_rear_d_time
            pev.eptcschargingstarttime = pev.ept_arrtime + ept_WT
            pev.ept_charging_duration = ept_charduration
            pev.cs = cs
            # pev.ept_totalcost = weight
            ept_diff_ch = cs.recieve_request(pev)


            # reward = 0
            # reward = -weight/60 - ept_diff_ch/60

            reward = -weight/60
            # reward = -pev.ept_totaldrivingtime / 60 - pev.ept_charging_duration / 60
            # reward = - ept_diff_ch/60
            # reward = -pev.ept_waitingtime
            # reward = -pev.ept_totaldrivingtime/60
            # reward = -(ept_WT/60*UNITtimecost + ept_cs_charging_cost)/10

            done = 0

            # print(pev.id, reward)

        if pev.id+1<setting.N_REQ:
            next_pev = self.request_be_EV[pev.id+1]
            # self.path_info = ta.get_feature_state_DQN_fleet(next_pev.t_start, next_pev, self.CS_list, self.graph, 0)

            self.path_info = ta.get_feature_state_DQN_time_fleet(next_pev.t_start, next_pev, self.CS_list, self.graph, 0)

            # next_state = [next_pev.curr_location/self.graph.num_node, next_pev.destination / self.graph.num_node, next_pev.curr_SOC, next_pev.req_SOC, (next_pev.t_start/60)/288, (next_pev.t_start - pev.t_start)/60] # , (next_pev.t_start/60)/288

            next_state = list(np.eye(34)[next_pev.curr_location - 1])
            next_state += list(np.eye(34)[next_pev.destination - 1])
            next_state += [next_pev.curr_SOC, (next_pev.t_start/60)/288, (next_pev.t_start - pev.t_start)/60] # , (next_pev.t_start/60)/288
            # next_state = [next_pev.curr_SOC] # , (next_pev.t_start/60)/288

            ept_WT_list = []
            for path in self.path_info:
                (cs, weight, chargingprice, ept_driving_cost, front_path, rear_path, front_path_distance, rear_path_distance,
                 ept_front_d_time, ept_rear_d_time, fpath_weight, rpath_weight, ept_WT, ept_charduration,
                 ept_cs_charging_cost, ept_home_charging_cost, ept_arrtime, tmp_avail) = path

                for av in tmp_avail:
                    cpstate = 0
                    if av<next_pev.curr_time:
                        cpstate=0
                    else:
                        cpstate = (av-next_pev.curr_time)/60
                    next_state += [cpstate]
                    ept_WT_list.append(cpstate)

                next_state += [ept_WT/60, ept_charduration/60, ept_front_d_time/60 , ept_rear_d_time/60, front_path_distance/20, rear_path_distance/20]  #, (ept_arrtime/60)/288

            # next_state = np.reshape(next_state, [1, self.state_size])

            return next_state, next_pev, reward, done

        else:

            for cs in self.CS_list:
                cs.sim_finish(self.graph)

            done = 0

            return np.zeros(self.state_size), -1, reward, done





def init_request(num_request, graph, seed):

    # if seed:
    #     print('Seed', seed)
    #     np.random.seed(seed)
    #     random.seed(seed)


    request_EV = []
    request_EV_unknown = []

    source_node_list = list(graph.source_node_set)
    destination_node_list = list(graph.destination_node_set)

    num_request_unknown = int(num_request*0.05)

    request_time = np.random.uniform(360, 1200, num_request)  # 06:00 ~ 20:00

    request_time.sort()
    # print(request_time)
    # request_time_unknown.sort()

    for i in range(num_request):
        s = source_node_list[np.random.randint(0, len(source_node_list) - 1)]
        while s in graph.cs_info.keys():
            s = source_node_list[np.random.randint(0, len(source_node_list) - 1)]

        d = destination_node_list[np.random.randint(0, len(destination_node_list) - 1)]
        while d in graph.cs_info.keys() or s==d:
            d = destination_node_list[np.random.randint(0, len(destination_node_list) - 1)]

        soc = np.random.uniform(0.20, 0.40)
        # soc = 0.3
        # req_soc = np.random.uniform(0.7, 0.9)
        req_soc = 0.9
        t_start = request_time[i]
        request_EV.append(EV(i, s, d, soc, req_soc, t_start))

    # for i in range(num_request_unknown):
    #     s = source_node_list[np.random.randint(0, len(source_node_list) - 1)]
    #     while s in graph.cs_info.keys():
    #         s = source_node_list[np.random.randint(0, len(source_node_list) - 1)]
    #
    #     d = destination_node_list[np.random.randint(0, len(destination_node_list) - 1)]
    #     while d in graph.cs_info.keys():
    #         d = destination_node_list[np.random.randint(0, len(destination_node_list) - 1)]
    #
    #     soc = np.random.uniform(0.25, 0.40)
    #     req_soc = 0.9
    #     t_start = request_time_unknown[i]
    #     request_EV_unknown.append(EV(i, s, d, soc, req_soc, t_start))


    # for ev in request_EV:
    #     print(ev.source, end=' ')
    # print()
    #
    # for ev in request_EV:
    #     print(ev.destination, end=' ')
    # print()

    # for ev in request_EV:
    #     print(ev.init_SOC, end=' ')
    # print()

    return request_EV


def createFolder(directory):
    try:
        if not os.path.isdir(directory):
            os.makedirs(directory)
    except OSError:
        print('error')
