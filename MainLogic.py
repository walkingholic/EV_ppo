import matplotlib.pyplot as plt
import heapq
import os
from haversine import haversine

EPISODES = 10000
N_REQ = 80
N_SOCKET = 2
EPS_DC = 0.9999
UNITtimecost = 8
ECRate = 0.16
Step_SOC = 0.15
Base_SOC = 0.6
Final_SOC = 0.9
N_SOC = int((Final_SOC-Base_SOC)/Step_SOC)+1
NCS= 5


def createFolder(directory):
    try:
        if not os.path.isdir(directory):
            os.makedirs(directory)
    except OSError:
        print('error')

class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):

        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]


def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b

    return abs(x1 - x2) + abs(y1 - y2)

def dijkstra_search(graph, start, goal):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while not frontier.empty():
        current = frontier.get()

        if current == goal:
            break

        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.weight(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost
                frontier.put(next, priority)
                came_from[next] = current

    return came_from, cost_so_far

def a_star_search(graph, start, goal):

    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while not frontier.empty():
        current = frontier.get()
        # print('frontier.get()', current)
        if current == goal:
            break
        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.weight(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic_astar(graph.nodes_xy(goal), graph.nodes_xy(next))
                frontier.put(next, priority)
                # print('frontier.put()', next, priority)
                came_from[next] = current

    return came_from, cost_so_far

def a_star_search_optimal(graph, start, goal):

    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while not frontier.empty():
        current = frontier.get()
        # print('frontier.get()', current)
        if current == goal:
            break
        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.weight(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic_astar(graph.nodes_xy(goal), graph.nodes_xy(next))
                frontier.put(next, priority)
                # print('frontier.put()', next, priority)
                came_from[next] = current

    return came_from, cost_so_far


def update_envir_weight(cs, graph, sim_time):
    # print(int(sim_time / 5))

    time_idx = int(sim_time / 5)
    for l_id in graph.link_data.keys():
        velo = graph.traffic_info[l_id][time_idx]
        # print(graph.link_data[l_id]['LENGTH'], velo)
        croad = ECRate*graph.link_data[l_id]['LENGTH']*cs.TOU_price[int(time_idx/ 12)]
        troad = graph.link_data[l_id]['LENGTH']/velo*UNITtimecost
        graph.link_data[l_id]['WEIGHT'] = croad+troad
def update_envir_weight_shortestpath(cs, graph, sim_time):
    # print(int(sim_time / 5))

    time_idx = int(sim_time / 5)
    for l_id in graph.link_data.keys():
        velo = graph.traffic_info[l_id][time_idx]

        # print(graph.link_data[l_id]['LENGTH'], velo)
        # croad = ECRate*graph.link_data[l_id]['LENGTH']*cs.TOU_price[int(time_idx/ 12)]
        # troad = graph.link_data[l_id]['LENGTH']/velo*UNITtimecost
        graph.link_data[l_id]['WEIGHT'] = graph.link_data[l_id]['LENGTH']
def update_envir_weight_shortesttime(cs, graph, sim_time):
    # print(int(sim_time / 5))

    time_idx = int(sim_time / 5)
    for l_id in graph.link_data.keys():
        velo = graph.traffic_info[l_id][time_idx]

        # print(graph.link_data[l_id]['LENGTH'], velo)
        # croad = ECRate*graph.link_data[l_id]['LENGTH']*cs.TOU_price[int(time_idx/ 12)]
        # troad = graph.link_data[l_id]['LENGTH']/velo*UNITtimecost
        graph.link_data[l_id]['WEIGHT'] = graph.link_data[l_id]['LENGTH'] / velo


def heuristic_astar(a, b):
    (x1, y1) = a
    (x2, y2) = b

    x1, y1 = a
    x2, y2 = b

    dist = haversine((y1, x1), (y2, x2), unit='km')
    return dist
def reconstruct_path(came_from, start, goal):
    current = goal
    path = []
    while current != start:
        path.append(current)
        # print('path.append(current)', current)
        current = came_from[current]
    path.append(start) # optional
    path.reverse() # optional
    return path

def update_ev(pev, simple_graph, fnode, tnode, sim_time):

    # if sim_time >= 1440:
    #     print('if sim time', sim_time)
    #     sim_time = sim_time%1440
    #     print('if sim time', sim_time)
    #     pev.curr_day += 1


    time_idx = int(sim_time%1440 / 5)
    # print(sim_time, time_idx)


    dist = simple_graph.distance(fnode, tnode)
    velo = simple_graph.velocity(fnode, tnode, time_idx)

    # print(' curTime: {}   curLoca: {}    sim: {}  velo: {}'.format(pev.curr_time, pev.curr_location, sim_time, velo))

    time_diff = (dist/velo)*60
    # pev.traveltime = pev.traveltime + time_diff
    pev.totaldrivingtime += time_diff
    pev.totaldrivingdistance += dist
    soc_before = pev.curr_SOC
    pev.curr_SOC = pev.curr_SOC - (dist * pev.ECRate) / pev.maxBCAPA
    pev.totalenergyconsumption = pev.totalenergyconsumption + dist * pev.ECRate

    new_sim_time = sim_time + time_diff

    time_idx = int(new_sim_time%1440 / 5)
    # print("fnode {} tnode {} dist {} velo {} soc_b {} soc_a {}".format(fnode, tnode, dist, velo, soc_before, pev.SOC))

    if pev.charged != 1:
        pev.fdist += dist
    else:
        pev.rdist += dist
    pev.curr_time = new_sim_time
    pev.curr_location = tnode
    # pev.path.append(pev.curr_location)
    return new_sim_time, time_diff
def pruning_evcs(pev, CS_list,graph, ncandi):
    cslist = []
    for cs in CS_list:
        x1, y1 = graph.nodes_xy(pev.curr_location)
        x2, y2 = graph.nodes_xy(cs.id)
        x3, y3 = graph.nodes_xy(pev.destination)

        f = haversine((y1, x1),(y2, x2), unit='km')
        r = haversine((y2, x2),(y3, x3), unit='km')
        airdist = f+r
        cslist.append((cs, airdist))
    cslist.sort(key=lambda element:element[1])
    return cslist[:ncandi]
def get_feature_state(sim_time, pev, CS_list, graph, ncandi):

    evcango = pev.curr_SOC * pev.maxBCAPA / pev.ECRate
    start = pev.curr_location
    end = pev.destination
    info = []

    for cs in CS_list:
        update_envir_weight(cs, graph, sim_time)
        evcs_id = cs.id
        came_from, cost_so_far = dijkstra_search(graph, start, evcs_id)
        front_path = reconstruct_path(came_from, start, evcs_id)
        front_path_distance = graph.get_path_distance(front_path)
        came_from, cost_so_far = dijkstra_search(graph, evcs_id, end)
        rear_path = reconstruct_path(came_from, evcs_id, end)
        rear_path_distance = graph.get_path_distance(rear_path)
        final_path = front_path + rear_path[1:]
        total_distance = graph.get_path_distance(final_path)
        fpath_weight = graph.get_path_weight(front_path)
        rpath_weight = graph.get_path_weight(rear_path)
        rear_consump_energy = rear_path_distance * pev.ECRate
        front_d_time = graph.get_path_drivingtime(front_path, int(sim_time / 5))
        rear_d_time = graph.get_path_drivingtime(rear_path, int(sim_time / 5))
        remainenergy = pev.maxBCAPA*pev.init_SOC - front_path_distance * pev.ECRate
        waiting_time = cs.waittime[int(sim_time / 5)]

        driving_cost = graph.get_path_weight(final_path)

        for i in range(0, N_SOC):
            req_soc = i*Step_SOC+Base_SOC
            charging_energy = pev.maxBCAPA*req_soc - remainenergy
            if charging_energy<=0:
                print('charging_energy error')
                input()
            chargingprice = cs.price[int(sim_time / 5)]
            charging_time = (charging_energy/(cs.chargingpower*pev.charging_effi))
            cs_charging_cost = charging_energy * chargingprice+charging_time*UNITtimecost
            athome_remainE = (pev.maxBCAPA*req_soc-rear_consump_energy)
            athome_soc = athome_remainE/pev.maxBCAPA
            home_charging_cost = (pev.maxBCAPA*pev.final_soc-athome_remainE)*cs.TOU_price[int(sim_time/60)]

            info.append((cs, req_soc, driving_cost, front_path, rear_path, front_path_distance, rear_path_distance,
                         front_d_time, rear_d_time, fpath_weight, rpath_weight, waiting_time,
                         charging_time, cs_charging_cost, home_charging_cost))

            # print('actions: ',cs.id, req_soc)

    return info
def get_true_arrtime(pev, cs, graph):
    # print('get_true_arrtime', pev.id)
    for i in range(len(pev.front_path) - 1):
        fnode = pev.front_path[i]
        tnode = pev.front_path[i + 1]

        _, time = update_ev(pev, graph, pev.curr_location, tnode, pev.curr_time)

        if pev.curr_SOC <= 0.0:
            print('No soc')
            input()
    # print(' curTime: {}   curLoca: {} '.format(pev.curr_time, pev.curr_location))

    return pev.curr_time
def finishi_trip(pev, cp, graph):
    # print('fin ev', pev.id, pev.cs.id, cp.id, pev.curr_time, pev.true_waitingtime)
    # print(pev.curr_location, pev.rear_path)
    for i in range(len(pev.rear_path) - 1):
        fnode = pev.rear_path[i]
        tnode = pev.rear_path[i + 1]

        _, time = update_ev(pev, graph, pev.curr_location, tnode, pev.curr_time)

        if pev.curr_SOC <= 0.0:
            print('finishi_trip No soc')
            input()
    # print(' curTime: {}   curLoca: {} '.format(pev.curr_time, pev.curr_location))

    pev.cschargingcost = pev.cscharingenergy * pev.cschargingprice
    pev.expense_time_part = (pev.totaldrivingtime + pev.true_waitingtime + pev.true_charging_duration)/60 * UNITtimecost
    pev.expense_cost_part = pev.totaldrivingdistance * pev.ECRate * pev.cschargingprice + pev.cschargingcost

    pev.totalcost = pev.expense_time_part + pev.expense_cost_part

    return pev.curr_time

def get_feature_state_DQN_time_fleet(cur_time, pev, CS_list, graph, ncandi):

    evcango = pev.init_SOC * pev.maxBCAPA / pev.ECRate
    start = pev.curr_location
    end = pev.destination
    info = []

    for cs in CS_list:
        update_envir_weight_shortesttime(cs, graph, cur_time) # 유닛 h
        evcs_id = cs.id
        came_from, cost_so_far = dijkstra_search(graph, start, evcs_id)
        front_path = reconstruct_path(came_from, start, evcs_id)
        front_path_distance = graph.get_path_distance(front_path)
        came_from, cost_so_far = dijkstra_search(graph, evcs_id, end)
        rear_path = reconstruct_path(came_from, evcs_id, end)
        rear_path_distance = graph.get_path_distance(rear_path)
        final_path = front_path + rear_path[1:]
        total_distance = graph.get_path_distance(final_path)

        fpath_weight = graph.get_path_weight(front_path)
        rpath_weight = graph.get_path_weight(rear_path)

        rear_consump_energy = rear_path_distance * pev.ECRate
        front_d_time = graph.get_path_drivingtime(front_path, int(cur_time / 5))*60
        rear_d_time = graph.get_path_drivingtime(rear_path, int(cur_time / 5))*60
        remainenergy = pev.maxBCAPA*pev.init_SOC - front_path_distance * pev.ECRate

        total_d_time = front_d_time+rear_d_time

        ept_arrtime = cur_time+front_d_time
        # print(cur_time, ept_arrtime)

        driving_cost = graph.get_path_weight(final_path)
        charging_energy = pev.maxBCAPA*pev.req_SOC - remainenergy

        if charging_energy<=0:
            print('charging_energy error')
            input()

        chargingprice = cs.price[int(cur_time / 5)]
        ept_charging_duration = (charging_energy/(cs.fastchargingpower*pev.charging_effi))*60


        athome_remainE = (pev.maxBCAPA*pev.req_SOC - rear_consump_energy)
        athome_soc = athome_remainE/pev.maxBCAPA
        home_charging_cost = (pev.maxBCAPA*pev.final_soc-athome_remainE)*cs.TOU_price[int(cur_time/60)]

        ept_WT, at_list, diff = cs.get_ept_WT(pev,ept_arrtime,  ept_charging_duration, cur_time, graph)
        # ept_WT, at_list, diff = cs.get_current_WT(pev, cur_time, graph)
        cs_charging_cost = charging_energy * chargingprice + (ept_charging_duration / 60 + ept_WT/60) * UNITtimecost

        weight = ept_WT + ept_charging_duration + total_d_time
        info.append((cs, weight, chargingprice, driving_cost, front_path, rear_path, front_path_distance, rear_path_distance,
                     front_d_time, rear_d_time, fpath_weight, rpath_weight, ept_WT,
                     ept_charging_duration, cs_charging_cost, home_charging_cost, ept_arrtime, at_list))

        # print('actions: ',cs.id, req_soc)

    return info
def get_feature_state_DQN_fleet(cur_time, pev, CS_list, graph, ncandi):

    evcango = pev.init_SOC * pev.maxBCAPA / pev.ECRate
    start = pev.curr_location
    end = pev.destination
    info = []

    for cs in CS_list:
        update_envir_weight(cs, graph, cur_time)
        evcs_id = cs.id
        came_from, cost_so_far = dijkstra_search(graph, start, evcs_id)
        front_path = reconstruct_path(came_from, start, evcs_id)
        front_path_distance = graph.get_path_distance(front_path)
        came_from, cost_so_far = dijkstra_search(graph, evcs_id, end)
        rear_path = reconstruct_path(came_from, evcs_id, end)
        rear_path_distance = graph.get_path_distance(rear_path)
        final_path = front_path + rear_path[1:]
        total_distance = graph.get_path_distance(final_path)

        fpath_weight = graph.get_path_weight(front_path)
        rpath_weight = graph.get_path_weight(rear_path)

        rear_consump_energy = rear_path_distance * pev.ECRate
        front_d_time = graph.get_path_drivingtime(front_path, int(cur_time / 5))*60
        rear_d_time = graph.get_path_drivingtime(rear_path, int(cur_time / 5))*60
        remainenergy = pev.maxBCAPA*pev.init_SOC - front_path_distance * pev.ECRate

        ept_arrtime = cur_time+front_d_time
        # print(cur_time, ept_arrtime)

        driving_cost = graph.get_path_weight(final_path)
        charging_energy = pev.maxBCAPA*pev.req_SOC - remainenergy

        if charging_energy<=0:
            print('charging_energy error')
            input()

        chargingprice = cs.price[int(cur_time / 5)]
        ept_charging_duration = (charging_energy/(cs.chargingpower*pev.charging_effi))*60


        athome_remainE = (pev.maxBCAPA*pev.req_SOC - rear_consump_energy)
        athome_soc = athome_remainE/pev.maxBCAPA
        home_charging_cost = (pev.maxBCAPA*pev.final_soc-athome_remainE)*cs.TOU_price[int(cur_time/60)]

        # ept_WT, at_list, diff = cs.get_ept_WT(pev,ept_arrtime,  ept_charging_duration, cur_time, graph)
        ept_WT, at_list, diff = cs.get_current_WT(pev, cur_time, graph)
        cs_charging_cost = charging_energy * chargingprice + (ept_charging_duration / 60 + ept_WT/60) * UNITtimecost

        weight = cs_charging_cost+driving_cost
        info.append((cs, weight, chargingprice, driving_cost, front_path, rear_path, front_path_distance, rear_path_distance,
                     front_d_time, rear_d_time, fpath_weight, rpath_weight, ept_WT,
                     ept_charging_duration, cs_charging_cost, home_charging_cost, ept_arrtime))

        # print('actions: ',cs.id, req_soc)

    return info
def get_feature_state_greedy_fleet(cur_time, pev, CS_list, graph, ncandi):
    evcango = pev.init_SOC * pev.maxBCAPA / pev.ECRate
    start = pev.curr_location
    end = pev.destination
    info = []

    for cs in CS_list:
        update_envir_weight(cs, graph, cur_time)
        evcs_id = cs.id
        came_from, cost_so_far = dijkstra_search(graph, start, evcs_id)
        front_path = reconstruct_path(came_from, start, evcs_id)
        front_path_distance = graph.get_path_distance(front_path)
        came_from, cost_so_far = dijkstra_search(graph, evcs_id, end)
        rear_path = reconstruct_path(came_from, evcs_id, end)
        rear_path_distance = graph.get_path_distance(rear_path)
        final_path = front_path + rear_path[1:]
        total_distance = graph.get_path_distance(final_path)

        if evcango>front_path_distance:
            fpath_weight = graph.get_path_weight(front_path)
            rpath_weight = graph.get_path_weight(rear_path)

            rear_consump_energy = rear_path_distance * pev.ECRate
            front_d_time = graph.get_path_drivingtime(front_path, int(cur_time / 5)) * 60
            rear_d_time = graph.get_path_drivingtime(rear_path, int(cur_time / 5)) * 60
            remainenergy = pev.maxBCAPA * pev.init_SOC - front_path_distance * pev.ECRate

            ept_arrtime = cur_time + front_d_time
            # print(cur_time, ept_arrtime)

            driving_cost = graph.get_path_weight(final_path)
            charging_energy = pev.maxBCAPA * pev.req_SOC - remainenergy

            if charging_energy <= 0:
                print('charging_energy error')
                input()

            chargingprice = cs.price[int(cur_time / 5)]
            ept_charging_duration = (charging_energy / (cs.chargingpower * pev.charging_effi)) * 60


            athome_remainE = (pev.maxBCAPA * pev.req_SOC - rear_consump_energy)
            athome_soc = athome_remainE / pev.maxBCAPA
            home_charging_cost = (pev.maxBCAPA * pev.final_soc - athome_remainE) * cs.TOU_price[int(cur_time / 60)]

            ept_WT, at_list, diff = cs.get_ept_WT(pev,ept_arrtime,  ept_charging_duration, cur_time, graph)
            # ept_WT, at_list, diff = cs.get_current_WT(pev, cur_time, graph)

            cs_charging_cost = charging_energy * chargingprice + (ept_charging_duration / 60 + ept_WT/60) * UNITtimecost

            weight = cs_charging_cost + driving_cost+ ept_WT
            info.append(
                (cs, weight, chargingprice, driving_cost, front_path, rear_path, front_path_distance, rear_path_distance,
                 front_d_time, rear_d_time, fpath_weight, rpath_weight, ept_WT,
                 ept_charging_duration, cs_charging_cost, home_charging_cost, ept_arrtime))
            # print('actions: ',cs.id, req_soc)

    return info
def get_greedy_time_cost_fleet(request_be_EV, CS_list, graph):


    for pev in request_be_EV:

        # charging_energy = pev.maxBCAPA * (pev.req_SOC - pev.curr_SOC)
        # charging_duration = (charging_energy / (60 * pev.charging_effi))
        # pev.ept_charging_duration = charging_duration * 60

        # print('=====================================================================')
        # print('be ID:{0:3} S:{1:3} D:{2:3} CurSOC:{3:0.2f} ReqSOC:{4:0.2f} Tstart:{5:0.2f} Tarr:{6:0.2f}'
        #       .format(pev.id, pev.source, pev.destination, pev.curr_SOC, pev.req_SOC, pev.t_start, pev.ept_arrtime))
        candi = []
        candi = get_feature_state_greedy_fleet(pev.t_start, pev, CS_list, graph, 0)

        # for cs, _, _, _, _, _, _, _, _, _, _, eptWT, ept_charduration, _, _, ept_arrtime in candi:
        #     print('ID: {0:3}  eptWT: {1:.2f}   eptCharduration: {3:.2f}   Len_reserv: {2:.2f}'.format(cs.id, eptWT, len(
        #         cs.reserve_ev), ept_charduration))

        candi.sort(key=lambda e: e[1])

        (cs, weight, chargingprice, ept_driving_cost, front_path, rear_path, front_path_distance, rear_path_distance, ept_front_d_time,
         ept_rear_d_time, fpath_weight, rpath_weight, ept_WT, ept_charduration, ept_cs_charging_cost,
         ept_home_charging_cost,
         ept_arrtime) = candi[0]

        pev.front_path = front_path
        pev.rear_path = rear_path
        pev.path = front_path + rear_path[1:]
        pev.ept_arrtime = ept_arrtime
        pev.true_arrtime = get_true_arrtime(pev, cs, graph)

        pev.ept_totaldrivingtime = ept_front_d_time + ept_rear_d_time
        pev.ept_waitingtime = ept_WT
        pev.ept_charging_duration = ept_charduration

        pev.cs = cs
        pev.cschargingprice = cs.price[int(pev.cschargingstarttime / 5)]
        pev.eptcschargingstarttime = pev.ept_arrtime + ept_WT
        cs.recieve_request(pev)

    for cs in CS_list:
        cs.sim_finish(graph)
        # print(cs.id, end=', ')
    # print('\n=====================================================================')

    tot_wt = 0
    tot_cost = 0
    for pev in request_be_EV:
        # print(
        #     'result,  ID: {0:3},  CSID: {1:3},  CurSOC: {2:.2f},  ReqSOC: {3:.2f},  Tstart: {4:5.2f},  EptTarr: {5:5.2f},  TruTarr: {10:5.2f},  diffTarr: {11:5.2f},  WT: {6:5.2f},  eptWT: {8:5.2f},  diffWT: {9:5.2f},  ChaStart: {7:5.2f},  finTime: {12:5.2f}'
        #     .format(pev.id, pev.cs.id, pev.curr_SOC, pev.req_SOC, pev.t_start, pev.ept_arrtime, pev.true_waitingtime,
        #             pev.cschargingstarttime, pev.ept_waitingtime, pev.time_diff_WT, pev.true_arrtime,
        #             pev.true_arrtime - pev.ept_arrtime, pev.curr_time))
        tot_wt += pev.true_waitingtime
        tot_cost += pev.totalcost

    # print('Avg. total waiting time: ', tot_wt / len(request_be_EV))
    # print('Total cost: ', tot_cost)

def get_feature_state_shortest_fleet(cur_time, pev, CS_list, graph, ncandi):

    evcango = pev.curr_SOC * pev.maxBCAPA / pev.ECRate
    start = pev.curr_location
    end = pev.destination
    info = []

    for cs in CS_list:
        update_envir_weight_shortestpath(cs, graph, cur_time)
        evcs_id = cs.id
        came_from, cost_so_far = dijkstra_search(graph, start, evcs_id)
        front_path = reconstruct_path(came_from, start, evcs_id)
        front_path_distance = graph.get_path_distance(front_path)
        came_from, cost_so_far = dijkstra_search(graph, evcs_id, end)
        rear_path = reconstruct_path(came_from, evcs_id, end)
        rear_path_distance = graph.get_path_distance(rear_path)
        final_path = front_path + rear_path[1:]
        total_distance = graph.get_path_distance(final_path)

        if evcango > front_path_distance:
            fpath_weight = graph.get_path_weight(front_path)
            rpath_weight = graph.get_path_weight(rear_path)
            rear_consump_energy = rear_path_distance * pev.ECRate
            front_d_time = graph.get_path_drivingtime(front_path, int(cur_time / 5))*60
            rear_d_time = graph.get_path_drivingtime(rear_path, int(cur_time / 5))*60
            remainenergy = pev.maxBCAPA*pev.init_SOC - front_path_distance * pev.ECRate

            ept_arrtime = cur_time+front_d_time
            # print(cur_time, ept_arrtime)

            time_idx = int(cur_time / 5)
            ept_driving_cost = total_distance*pev.ECRate*cs.TOU_price[int(time_idx / 12)]+(front_d_time+rear_d_time)* UNITtimecost


            charging_energy = pev.maxBCAPA*pev.req_SOC - remainenergy
            if charging_energy<=0:
                print('charging_energy error')
                input()

            chargingprice = cs.price[int(cur_time / 5)]
            ept_charging_duration = (charging_energy/(cs.fastchargingpower*pev.charging_effi))*60

            cs_charging_cost = charging_energy * chargingprice + ept_charging_duration*UNITtimecost

            athome_remainE = (pev.maxBCAPA*pev.req_SOC - rear_consump_energy)
            athome_soc = athome_remainE/pev.maxBCAPA
            home_charging_cost = (pev.maxBCAPA*pev.final_soc-athome_remainE)*cs.TOU_price[int(cur_time/60)]

            ept_WT, at_list, diff = cs.get_ept_WT(pev,ept_arrtime,  ept_charging_duration, cur_time, graph)
            # ept_WT, at_list, diff = cs.get_current_WT(pev, cur_time, graph)

            weight = total_distance
            info.append((cs, weight, ept_driving_cost, front_path, rear_path, front_path_distance, rear_path_distance,
                         front_d_time, rear_d_time, fpath_weight, rpath_weight, ept_WT,
                         ept_charging_duration, cs_charging_cost, home_charging_cost, ept_arrtime))

                # print('actions: ',cs.id, req_soc)

    return info
def get_greedy_shortest_fleet(request_be_EV, CS_list, graph):


    for pev in request_be_EV:

        # charging_energy = pev.maxBCAPA * (pev.req_SOC - pev.curr_SOC)
        # charging_duration = (charging_energy / (60 * pev.charging_effi))
        # pev.ept_charging_duration = charging_duration * 60

        # print('=====================================================================')
        # print('be ID:{0:3} S:{1:3} D:{2:3} CurSOC:{3:0.2f} ReqSOC:{4:0.2f} Tstart:{5:0.2f} Tarr:{6:0.2f}'
        #       .format(pev.id, pev.source, pev.destination, pev.curr_SOC, pev.req_SOC, pev.t_start, pev.ept_arrtime))
        candi = []
        candi = get_feature_state_shortest_fleet(pev.t_start, pev, CS_list, graph, 0)

        # for cs, _, _, _, _, _, _, _, _, _, _, eptWT, ept_charduration, _, _, ept_arrtime in candi:
        #     print('ID: {0:3}  eptWT: {1:.2f}   eptCharduration: {3:.2f}   Len_reserv: {2:.2f}'.format(cs.id, eptWT, len(
        #         cs.reserve_ev), ept_charduration))

        candi.sort(key=lambda e: e[1])

        (cs, weight, ept_driving_cost, front_path, rear_path, front_path_distance, rear_path_distance, ept_front_d_time,
         ept_rear_d_time, fpath_weight, rpath_weight, ept_WT, ept_charduration, ept_cs_charging_cost,
         ept_home_charging_cost,
         ept_arrtime) = candi[0]

        pev.front_path = front_path
        pev.rear_path = rear_path
        pev.path = front_path + rear_path[1:]
        pev.ept_arrtime = ept_arrtime
        pev.true_arrtime = get_true_arrtime(pev, cs, graph)

        pev.ept_totaldrivingtime = ept_front_d_time + ept_rear_d_time
        pev.ept_waitingtime = ept_WT
        pev.ept_charging_duration = ept_charduration

        pev.cs = cs
        pev.cschargingprice = cs.price[int(pev.cschargingstarttime / 5)]
        pev.eptcschargingstarttime = pev.ept_arrtime + ept_WT
        cs.recieve_request(pev)

    for cs in CS_list:
        cs.sim_finish(graph)
        # print(cs.id, end=', ')
    # print('\n=====================================================================')

    tot_wt = 0
    tot_cost = 0
    for pev in request_be_EV:
        # print(
        #     'result,  ID: {0:3},  CSID: {1:3},  CurSOC: {2:.2f},  ReqSOC: {3:.2f},  Tstart: {4:5.2f},  EptTarr: {5:5.2f},  TruTarr: {10:5.2f},  diffTarr: {11:5.2f},  WT: {6:5.2f},  eptWT: {8:5.2f},  diffWT: {9:5.2f},  ChaStart: {7:5.2f},  finTime: {12:5.2f}'
        #     .format(pev.id, pev.cs.id, pev.curr_SOC, pev.req_SOC, pev.t_start, pev.ept_arrtime, pev.true_waitingtime,
        #             pev.cschargingstarttime, pev.ept_waitingtime, pev.time_diff_WT, pev.true_arrtime,
        #             pev.true_arrtime - pev.ept_arrtime, pev.curr_time))
        tot_wt += pev.true_waitingtime
        tot_cost += pev.totalcost

    # print('Avg. total waiting time: ', tot_wt / len(request_be_EV))
    # print('Total cost: ', tot_cost)
def get_feature_state_shorttime_fleet_noreserve(cur_time, pev, CS_list, graph, ncandi):
    evcango = pev.curr_SOC * pev.maxBCAPA / pev.ECRate
    start = pev.curr_location
    end = pev.destination
    info = []

    for cs in CS_list:
        update_envir_weight_shortesttime(cs, graph, cur_time)
        evcs_id = cs.id
        came_from, cost_so_far = dijkstra_search(graph, start, evcs_id)
        front_path = reconstruct_path(came_from, start, evcs_id)
        front_path_distance = graph.get_path_distance(front_path)
        came_from, cost_so_far = dijkstra_search(graph, evcs_id, end)
        rear_path = reconstruct_path(came_from, evcs_id, end)
        rear_path_distance = graph.get_path_distance(rear_path)
        final_path = front_path + rear_path[1:]
        total_distance = graph.get_path_distance(final_path)

        if evcango > front_path_distance:

            fpath_weight = graph.get_path_weight(front_path)
            rpath_weight = graph.get_path_weight(rear_path)

            rear_consump_energy = rear_path_distance * pev.ECRate
            front_d_time = graph.get_path_drivingtime(front_path, int(cur_time / 5)) * 60
            rear_d_time = graph.get_path_drivingtime(rear_path, int(cur_time / 5)) * 60
            remainenergy = pev.maxBCAPA * pev.init_SOC - front_path_distance * pev.ECRate

            ept_arrtime = cur_time + front_d_time
            # print(cur_time, ept_arrtime)

            time_idx = int(cur_time / 5)
            ept_driving_cost = total_distance * pev.ECRate * cs.TOU_price[int(time_idx / 12)] + (
                        front_d_time + rear_d_time) * UNITtimecost

            charging_energy = pev.maxBCAPA * pev.req_SOC - remainenergy

            if charging_energy <= 0:
                print('charging_energy error')
                input()

            chargingprice = cs.price[int(cur_time / 5)]
            ept_charging_duration = (charging_energy / (cs.chargingpower * pev.charging_effi)) * 60

            cs_charging_cost = charging_energy * chargingprice + ept_charging_duration/60 * UNITtimecost

            athome_remainE = (pev.maxBCAPA * pev.req_SOC - rear_consump_energy)
            athome_soc = athome_remainE / pev.maxBCAPA
            home_charging_cost = (pev.maxBCAPA * pev.final_soc - athome_remainE) * cs.TOU_price[int(cur_time / 60)]

            # ept_WT, at_list, diff = cs.get_ept_WT(pev,ept_arrtime,  ept_charging_duration, cur_time, graph)
            ept_WT, at_list, diff = cs.get_current_WT(pev, cur_time, graph)

            weight = graph.get_path_weight(final_path)*60 + ept_WT + ept_charging_duration
            info.append((cs, weight, ept_driving_cost, front_path, rear_path, front_path_distance, rear_path_distance,
                         front_d_time, rear_d_time, fpath_weight, rpath_weight, ept_WT,
                         ept_charging_duration, cs_charging_cost, home_charging_cost, ept_arrtime))

            # print('actions: ',cs.id, req_soc)

    return info
def get_greedy_shorttime_fleet_noreserve(request_be_EV, CS_list, graph):
    for pev in request_be_EV:
        # charging_energy = pev.maxBCAPA * (pev.req_SOC - pev.curr_SOC)
        # charging_duration = (charging_energy / (60 * pev.charging_effi))
        # pev.ept_charging_duration = charging_duration * 60

        # print('=====================================================================')
        # print('be ID:{0:3} S:{1:3} D:{2:3} CurSOC:{3:0.2f} ReqSOC:{4:0.2f} Tstart:{5:0.2f} Tarr:{6:0.2f}'
        #       .format(pev.id, pev.source, pev.destination, pev.curr_SOC, pev.req_SOC, pev.t_start, pev.ept_arrtime))
        candi = []
        candi = get_feature_state_shorttime_fleet_noreserve(pev.t_start, pev, CS_list, graph, 0)

        # for cs, _, _, _, _, _, _, _, _, _, _, eptWT, ept_charduration, _, _, ept_arrtime in candi:
        #     print('ID: {0:3}  eptWT: {1:.2f}   eptCharduration: {3:.2f}   Len_reserv: {2:.2f}'.format(cs.id, eptWT, len(
        #         cs.reserve_ev), ept_charduration))

        candi.sort(key=lambda e: e[1])

        (cs, weight, ept_driving_cost, front_path, rear_path, front_path_distance, rear_path_distance, ept_front_d_time,
         ept_rear_d_time, fpath_weight, rpath_weight, ept_WT, ept_charduration, ept_cs_charging_cost,
         ept_home_charging_cost,
         ept_arrtime) = candi[0]

        pev.front_path = front_path
        pev.rear_path = rear_path
        pev.path = front_path + rear_path[1:]
        pev.ept_arrtime = ept_arrtime
        pev.true_arrtime = get_true_arrtime(pev, cs, graph)

        pev.ept_waitingtime = ept_WT
        pev.ept_totaldrivingtime = ept_front_d_time + ept_rear_d_time
        pev.ept_charging_duration = ept_charduration

        pev.cs = cs
        pev.cschargingprice = cs.price[int(pev.cschargingstarttime / 5)]
        pev.eptcschargingstarttime = pev.ept_arrtime + ept_WT
        cs.recieve_request(pev)

    for cs in CS_list:
        cs.sim_finish(graph)
        # print(cs.id, end=', ')
    # print('\n=====================================================================')

    tot_wt = 0
    tot_cost = 0
    for pev in request_be_EV:
        # print(
        #     'result,  ID: {0:3},  CSID: {1:3},  CurSOC: {2:.2f},  ReqSOC: {3:.2f},  Tstart: {4:5.2f},  EptTarr: {5:5.2f},  TruTarr: {10:5.2f},  diffTarr: {11:5.2f},  WT: {6:5.2f},  eptWT: {8:5.2f},  diffWT: {9:5.2f},  ChaStart: {7:5.2f},  finTime: {12:5.2f}'
        #     .format(pev.id, pev.cs.id, pev.curr_SOC, pev.req_SOC, pev.t_start, pev.ept_arrtime, pev.true_waitingtime,
        #             pev.cschargingstarttime, pev.ept_waitingtime, pev.time_diff_WT, pev.true_arrtime,
        #             pev.true_arrtime - pev.ept_arrtime, pev.curr_time))
        tot_wt += pev.true_waitingtime
        tot_cost += pev.totalcost

    # print('Avg. total waiting time: ', tot_wt / len(request_be_EV))
    # print('Total cost: ', tot_cost)

def get_feature_state_shorttime_fleet_reserve(cur_time, pev, CS_list, graph, ncandi):
    evcango = pev.curr_SOC * pev.maxBCAPA / pev.ECRate
    start = pev.curr_location
    end = pev.destination
    info = []

    for cs in CS_list:
        update_envir_weight_shortesttime(cs, graph, cur_time)
        evcs_id = cs.id
        came_from, cost_so_far = dijkstra_search(graph, start, evcs_id)
        front_path = reconstruct_path(came_from, start, evcs_id)
        front_path_distance = graph.get_path_distance(front_path)
        came_from, cost_so_far = dijkstra_search(graph, evcs_id, end)
        rear_path = reconstruct_path(came_from, evcs_id, end)
        rear_path_distance = graph.get_path_distance(rear_path)
        final_path = front_path + rear_path[1:]
        total_distance = graph.get_path_distance(final_path)

        if evcango > front_path_distance:

            fpath_weight = graph.get_path_weight(front_path)
            rpath_weight = graph.get_path_weight(rear_path)

            rear_consump_energy = rear_path_distance * pev.ECRate
            front_d_time = graph.get_path_drivingtime(front_path, int(cur_time / 5)) * 60
            rear_d_time = graph.get_path_drivingtime(rear_path, int(cur_time / 5)) * 60
            remainenergy = pev.maxBCAPA * pev.init_SOC - front_path_distance * pev.ECRate

            ept_arrtime = cur_time + front_d_time
            # print(cur_time, ept_arrtime)

            time_idx = int(cur_time / 5)
            ept_driving_cost = total_distance * pev.ECRate * cs.TOU_price[int(time_idx / 12)] + (
                        front_d_time + rear_d_time) * UNITtimecost

            charging_energy = pev.maxBCAPA * pev.req_SOC - remainenergy

            if charging_energy <= 0:
                print('charging_energy error')
                input()

            chargingprice = cs.price[int(cur_time / 5)]
            ept_charging_duration = (charging_energy / (cs.fastchargingpower * pev.charging_effi)) * 60

            cs_charging_cost = charging_energy * chargingprice + ept_charging_duration/60 * UNITtimecost

            athome_remainE = (pev.maxBCAPA * pev.req_SOC - rear_consump_energy)
            athome_soc = athome_remainE / pev.maxBCAPA
            home_charging_cost = (pev.maxBCAPA * pev.final_soc - athome_remainE) * cs.TOU_price[int(cur_time / 60)]

            ept_WT, at_list, diff = cs.get_ept_WT(pev,ept_arrtime,  ept_charging_duration, cur_time, graph)
            # ept_WT, at_list, diff = cs.get_current_WT(pev, cur_time, graph)

            weight = graph.get_path_weight(final_path)*60 + ept_WT + ept_charging_duration
            info.append((cs, weight, ept_driving_cost, front_path, rear_path, front_path_distance, rear_path_distance,
                         front_d_time, rear_d_time, fpath_weight, rpath_weight, ept_WT,
                         ept_charging_duration, cs_charging_cost, home_charging_cost, ept_arrtime))

            # print('actions: ',cs.id, req_soc)

    return info
def get_greedy_shorttime_fleet_reserve(request_be_EV, CS_list, graph):
    for pev in request_be_EV:
        # charging_energy = pev.maxBCAPA * (pev.req_SOC - pev.curr_SOC)
        # charging_duration = (charging_energy / (60 * pev.charging_effi))
        # pev.ept_charging_duration = charging_duration * 60

        # print('=====================================================================')
        # print('be ID:{0:3} S:{1:3} D:{2:3} CurSOC:{3:0.2f} ReqSOC:{4:0.2f} Tstart:{5:0.2f} Tarr:{6:0.2f}'
        #       .format(pev.id, pev.source, pev.destination, pev.curr_SOC, pev.req_SOC, pev.t_start, pev.ept_arrtime))
        candi = []
        candi = get_feature_state_shorttime_fleet_reserve(pev.t_start, pev, CS_list, graph, 0)

        # for cs, _, _, _, _, _, _, _, _, _, _, eptWT, ept_charduration, _, _, ept_arrtime in candi:
        #     print('ID: {0:3}  eptWT: {1:.2f}   eptCharduration: {3:.2f}   Len_reserv: {2:.2f}'.format(cs.id, eptWT, len(
        #         cs.reserve_ev), ept_charduration))

        candi.sort(key=lambda e: e[1])

        (cs, weight, ept_driving_cost, front_path, rear_path, front_path_distance, rear_path_distance, ept_front_d_time,
         ept_rear_d_time, fpath_weight, rpath_weight, ept_WT, ept_charduration, ept_cs_charging_cost,
         ept_home_charging_cost,
         ept_arrtime) = candi[0]

        pev.front_path = front_path
        pev.rear_path = rear_path
        pev.path = front_path + rear_path[1:]
        pev.ept_arrtime = ept_arrtime
        pev.true_arrtime = get_true_arrtime(pev, cs, graph)

        pev.ept_waitingtime = ept_WT
        pev.ept_totaldrivingtime = ept_front_d_time + ept_rear_d_time
        pev.ept_charging_duration = ept_charduration

        pev.cs = cs
        pev.cschargingprice = cs.price[int(pev.cschargingstarttime / 5)]
        pev.eptcschargingstarttime = pev.ept_arrtime + ept_WT
        cs.recieve_request(pev)

    for cs in CS_list:
        cs.sim_finish(graph)
        # print(cs.id, end=', ')
    # print('\n=====================================================================')

    tot_wt = 0
    tot_cost = 0
    for pev in request_be_EV:
        # print(
        #     'result,  ID: {0:3},  CSID: {1:3},  CurSOC: {2:.2f},  ReqSOC: {3:.2f},  Tstart: {4:5.2f},  EptTarr: {5:5.2f},  TruTarr: {10:5.2f},  diffTarr: {11:5.2f},  WT: {6:5.2f},  eptWT: {8:5.2f},  diffWT: {9:5.2f},  ChaStart: {7:5.2f},  finTime: {12:5.2f}'
        #     .format(pev.id, pev.cs.id, pev.curr_SOC, pev.req_SOC, pev.t_start, pev.ept_arrtime, pev.true_waitingtime,
        #             pev.cschargingstarttime, pev.ept_waitingtime, pev.time_diff_WT, pev.true_arrtime,
        #             pev.true_arrtime - pev.ept_arrtime, pev.curr_time))
        tot_wt += pev.true_waitingtime
        tot_cost += pev.totalcost

    # print('Avg. total waiting time: ', tot_wt / len(request_be_EV))
    # print('Total cost: ', tot_cost)

def get_feature_state_shortcharging_distance_fleet_reserve(cur_time, pev, CS_list, graph, ncandi):
    evcango = pev.curr_SOC * pev.maxBCAPA / pev.ECRate
    start = pev.curr_location
    end = pev.destination
    info = []

    for cs in CS_list:
        update_envir_weight_shortesttime(cs, graph, cur_time)
        evcs_id = cs.id
        came_from, cost_so_far = dijkstra_search(graph, start, evcs_id)
        front_path = reconstruct_path(came_from, start, evcs_id)
        front_path_distance = graph.get_path_distance(front_path)
        came_from, cost_so_far = dijkstra_search(graph, evcs_id, end)
        rear_path = reconstruct_path(came_from, evcs_id, end)
        rear_path_distance = graph.get_path_distance(rear_path)
        final_path = front_path + rear_path[1:]
        total_distance = graph.get_path_distance(final_path)

        if evcango > front_path_distance:

            fpath_weight = graph.get_path_weight(front_path)
            rpath_weight = graph.get_path_weight(rear_path)

            rear_consump_energy = rear_path_distance * pev.ECRate
            front_d_time = graph.get_path_drivingtime(front_path, int(cur_time / 5)) * 60
            rear_d_time = graph.get_path_drivingtime(rear_path, int(cur_time / 5)) * 60
            remainenergy = pev.maxBCAPA * pev.init_SOC - front_path_distance * pev.ECRate

            ept_arrtime = cur_time + front_d_time
            # print(cur_time, ept_arrtime)

            time_idx = int(cur_time / 5)
            ept_driving_cost = total_distance * pev.ECRate * cs.TOU_price[int(time_idx / 12)] + (
                        front_d_time + rear_d_time) * UNITtimecost

            charging_energy = pev.maxBCAPA * pev.req_SOC - remainenergy

            if charging_energy <= 0:
                print('charging_energy error')
                input()

            chargingprice = cs.price[int(cur_time / 5)]
            ept_charging_duration = (charging_energy / (cs.chargingpower * pev.charging_effi)) * 60

            cs_charging_cost = charging_energy * chargingprice + ept_charging_duration/60 * UNITtimecost

            athome_remainE = (pev.maxBCAPA * pev.req_SOC - rear_consump_energy)
            athome_soc = athome_remainE / pev.maxBCAPA
            home_charging_cost = (pev.maxBCAPA * pev.final_soc - athome_remainE) * cs.TOU_price[int(cur_time / 60)]

            ept_WT, at_list, diff = cs.get_ept_WT(pev,ept_arrtime,  ept_charging_duration, cur_time, graph)
            # ept_WT, at_list, diff = cs.get_current_WT(pev, cur_time, graph)

            weight = graph.get_path_weight(final_path)*60 + ept_WT + ept_charging_duration
            info.append((cs, weight, ept_driving_cost, front_path, rear_path, front_path_distance, rear_path_distance,
                         front_d_time, rear_d_time, fpath_weight, rpath_weight, ept_WT,
                         ept_charging_duration, cs_charging_cost, home_charging_cost, ept_arrtime))

            # print('actions: ',cs.id, req_soc)

    return info
def get_greedy_shortcharging_distance_fleet_reserve(request_be_EV, CS_list, graph):
    for pev in request_be_EV:
        # charging_energy = pev.maxBCAPA * (pev.req_SOC - pev.curr_SOC)
        # charging_duration = (charging_energy / (60 * pev.charging_effi))
        # pev.ept_charging_duration = charging_duration * 60

        # print('=====================================================================')
        # print('be ID:{0:3} S:{1:3} D:{2:3} CurSOC:{3:0.2f} ReqSOC:{4:0.2f} Tstart:{5:0.2f} Tarr:{6:0.2f}'
        #       .format(pev.id, pev.source, pev.destination, pev.curr_SOC, pev.req_SOC, pev.t_start, pev.ept_arrtime))
        candi = []
        candi = get_feature_state_shortcharging_distance_fleet_reserve(pev.t_start, pev, CS_list, graph, 0)

        # for cs, _, _, _, _, _, _, _, _, _, _, eptWT, ept_charduration, _, _, ept_arrtime in candi:
        #     print('ID: {0:3}  eptWT: {1:.2f}   eptCharduration: {3:.2f}   Len_reserv: {2:.2f}'.format(cs.id, eptWT, len(
        #         cs.reserve_ev), ept_charduration))

        candi.sort(key=lambda e: e[1])

        (cs, weight, ept_driving_cost, front_path, rear_path, front_path_distance, rear_path_distance, ept_front_d_time,
         ept_rear_d_time, fpath_weight, rpath_weight, ept_WT, ept_charduration, ept_cs_charging_cost,
         ept_home_charging_cost,
         ept_arrtime) = candi[0]

        pev.front_path = front_path
        pev.rear_path = rear_path
        pev.path = front_path + rear_path[1:]
        pev.ept_arrtime = ept_arrtime
        pev.true_arrtime = get_true_arrtime(pev, cs, graph)

        pev.ept_waitingtime = ept_WT
        pev.ept_totaldrivingtime = ept_front_d_time + ept_rear_d_time
        pev.ept_charging_duration = ept_charduration

        pev.cs = cs
        pev.cschargingprice = cs.price[int(pev.cschargingstarttime / 5)]
        pev.eptcschargingstarttime = pev.ept_arrtime + ept_WT
        cs.recieve_request(pev)

    for cs in CS_list:
        cs.sim_finish(graph)
        # print(cs.id, end=', ')
    # print('\n=====================================================================')

    tot_wt = 0
    tot_cost = 0
    for pev in request_be_EV:
        # print(
        #     'result,  ID: {0:3},  CSID: {1:3},  CurSOC: {2:.2f},  ReqSOC: {3:.2f},  Tstart: {4:5.2f},  EptTarr: {5:5.2f},  TruTarr: {10:5.2f},  diffTarr: {11:5.2f},  WT: {6:5.2f},  eptWT: {8:5.2f},  diffWT: {9:5.2f},  ChaStart: {7:5.2f},  finTime: {12:5.2f}'
        #     .format(pev.id, pev.cs.id, pev.curr_SOC, pev.req_SOC, pev.t_start, pev.ept_arrtime, pev.true_waitingtime,
        #             pev.cschargingstarttime, pev.ept_waitingtime, pev.time_diff_WT, pev.true_arrtime,
        #             pev.true_arrtime - pev.ept_arrtime, pev.curr_time))
        tot_wt += pev.true_waitingtime
        tot_cost += pev.totalcost

    # print('Avg. total waiting time: ', tot_wt / len(request_be_EV))
    # print('Total cost: ', tot_cost)


def get_feature_state_shortwt_fleet_noreserv(cur_time, pev, CS_list, graph, ncandi):
    evcango = pev.curr_SOC * pev.maxBCAPA / pev.ECRate
    start = pev.curr_location
    end = pev.destination
    info = []

    for cs in CS_list:
        update_envir_weight_shortesttime(cs, graph, cur_time)
        evcs_id = cs.id
        came_from, cost_so_far = dijkstra_search(graph, start, evcs_id)
        front_path = reconstruct_path(came_from, start, evcs_id)
        front_path_distance = graph.get_path_distance(front_path)
        came_from, cost_so_far = dijkstra_search(graph, evcs_id, end)
        rear_path = reconstruct_path(came_from, evcs_id, end)
        rear_path_distance = graph.get_path_distance(rear_path)
        final_path = front_path + rear_path[1:]
        total_distance = graph.get_path_distance(final_path)

        if evcango > front_path_distance:

            fpath_weight = graph.get_path_weight(front_path)
            rpath_weight = graph.get_path_weight(rear_path)

            rear_consump_energy = rear_path_distance * pev.ECRate
            front_d_time = graph.get_path_drivingtime(front_path, int(cur_time / 5)) * 60
            rear_d_time = graph.get_path_drivingtime(rear_path, int(cur_time / 5)) * 60
            remainenergy = pev.maxBCAPA * pev.init_SOC - front_path_distance * pev.ECRate

            ept_arrtime = cur_time + front_d_time
            # print(cur_time, ept_arrtime)

            time_idx = int(cur_time / 5)
            ept_driving_cost = total_distance * pev.ECRate * cs.TOU_price[int(time_idx / 12)] + (
                        front_d_time + rear_d_time) * UNITtimecost

            charging_energy = pev.maxBCAPA * pev.req_SOC - remainenergy

            if charging_energy <= 0:
                print('charging_energy error')
                input()

            chargingprice = cs.price[int(cur_time / 5)]
            ept_charging_duration = (charging_energy / (cs.chargingpower * pev.charging_effi)) * 60

            cs_charging_cost = charging_energy * chargingprice + ept_charging_duration/60 * UNITtimecost

            athome_remainE = (pev.maxBCAPA * pev.req_SOC - rear_consump_energy)
            athome_soc = athome_remainE / pev.maxBCAPA
            home_charging_cost = (pev.maxBCAPA * pev.final_soc - athome_remainE) * cs.TOU_price[int(cur_time / 60)]

            # ept_WT, at_list, diff = cs.get_ept_WT(pev,ept_arrtime,  ept_charging_duration, cur_time, graph)
            ept_WT, at_list, diff = cs.get_current_WT(pev, cur_time, graph)

            weight = ept_WT
            info.append((cs, weight, ept_driving_cost, front_path, rear_path, front_path_distance, rear_path_distance,
                         front_d_time, rear_d_time, fpath_weight, rpath_weight, ept_WT,
                         ept_charging_duration, cs_charging_cost, home_charging_cost, ept_arrtime))

            # print('actions: ',cs.id, req_soc)

    return info

def get_greedy_shortwt_fleet_noreserv(request_be_EV, CS_list, graph):
    for pev in request_be_EV:
        # charging_energy = pev.maxBCAPA * (pev.req_SOC - pev.curr_SOC)
        # charging_duration = (charging_energy / (60 * pev.charging_effi))
        # pev.ept_charging_duration = charging_duration * 60

        # print('=====================================================================')
        # print('be ID:{0:3} S:{1:3} D:{2:3} CurSOC:{3:0.2f} ReqSOC:{4:0.2f} Tstart:{5:0.2f} Tarr:{6:0.2f}'
        #       .format(pev.id, pev.source, pev.destination, pev.curr_SOC, pev.req_SOC, pev.t_start, pev.ept_arrtime))
        candi = []
        candi = get_feature_state_shortwt_fleet_noreserv(pev.t_start, pev, CS_list, graph, 0)

        # for cs, _, _, _, _, _, _, _, _, _, _, eptWT, ept_charduration, _, _, ept_arrtime in candi:
        #     print('ID: {0:3}  eptWT: {1:.2f}   eptCharduration: {3:.2f}   Len_reserv: {2:.2f}'.format(cs.id, eptWT, len(
        #         cs.reserve_ev), ept_charduration))

        candi.sort(key=lambda e: e[1])

        (cs, weight, ept_driving_cost, front_path, rear_path, front_path_distance, rear_path_distance, ept_front_d_time,
         ept_rear_d_time, fpath_weight, rpath_weight, ept_WT, ept_charduration, ept_cs_charging_cost,
         ept_home_charging_cost,
         ept_arrtime) = candi[0]

        pev.front_path = front_path
        pev.rear_path = rear_path
        pev.path = front_path + rear_path[1:]
        pev.ept_arrtime = ept_arrtime
        pev.true_arrtime = get_true_arrtime(pev, cs, graph)
        pev.ept_waitingtime = ept_WT
        pev.ept_charging_duration = ept_charduration
        pev.ept_totaldrivingtime = ept_front_d_time + ept_rear_d_time
        pev.cs = cs
        pev.cschargingprice = cs.price[int(pev.cschargingstarttime / 5)]
        pev.eptcschargingstarttime = pev.ept_arrtime + ept_WT
        cs.recieve_request(pev)

    for cs in CS_list:
        cs.sim_finish(graph)
        # print(cs.id, end=', ')
    # print('\n=====================================================================')

    tot_wt = 0
    tot_cost = 0
    for pev in request_be_EV:
        # print(
        #     'result,  ID: {0:3},  CSID: {1:3},  CurSOC: {2:.2f},  ReqSOC: {3:.2f},  Tstart: {4:5.2f},  EptTarr: {5:5.2f},  TruTarr: {10:5.2f},  diffTarr: {11:5.2f},  WT: {6:5.2f},  eptWT: {8:5.2f},  diffWT: {9:5.2f},  ChaStart: {7:5.2f},  finTime: {12:5.2f}'
        #     .format(pev.id, pev.cs.id, pev.curr_SOC, pev.req_SOC, pev.t_start, pev.ept_arrtime, pev.true_waitingtime,
        #             pev.cschargingstarttime, pev.ept_waitingtime, pev.time_diff_WT, pev.true_arrtime,
        #             pev.true_arrtime - pev.ept_arrtime, pev.curr_time))
        tot_wt += pev.true_waitingtime
        tot_cost += pev.totalcost

    # print('Avg. total waiting time: ', tot_wt / len(request_be_EV))
    # print('Total cost: ', tot_cost)

def get_feature_state_shortwt_fleet_reserv(cur_time, pev, CS_list, graph, ncandi):
    evcango = pev.curr_SOC * pev.maxBCAPA / pev.ECRate
    start = pev.curr_location
    end = pev.destination
    info = []

    for cs in CS_list:
        update_envir_weight_shortesttime(cs, graph, cur_time)
        evcs_id = cs.id
        came_from, cost_so_far = dijkstra_search(graph, start, evcs_id)
        front_path = reconstruct_path(came_from, start, evcs_id)
        front_path_distance = graph.get_path_distance(front_path)
        came_from, cost_so_far = dijkstra_search(graph, evcs_id, end)
        rear_path = reconstruct_path(came_from, evcs_id, end)
        rear_path_distance = graph.get_path_distance(rear_path)
        final_path = front_path + rear_path[1:]
        total_distance = graph.get_path_distance(final_path)

        if evcango > front_path_distance:

            fpath_weight = graph.get_path_weight(front_path)
            rpath_weight = graph.get_path_weight(rear_path)

            rear_consump_energy = rear_path_distance * pev.ECRate
            front_d_time = graph.get_path_drivingtime(front_path, int(cur_time / 5)) * 60
            rear_d_time = graph.get_path_drivingtime(rear_path, int(cur_time / 5)) * 60
            remainenergy = pev.maxBCAPA * pev.init_SOC - front_path_distance * pev.ECRate

            ept_arrtime = cur_time + front_d_time
            # print(cur_time, ept_arrtime)

            time_idx = int(cur_time / 5)
            ept_driving_cost = total_distance * pev.ECRate * cs.TOU_price[int(time_idx / 12)] + (
                        front_d_time + rear_d_time) * UNITtimecost

            charging_energy = pev.maxBCAPA * pev.req_SOC - remainenergy

            if charging_energy <= 0:
                print('charging_energy error')
                input()

            chargingprice = cs.price[int(cur_time / 5)]
            ept_charging_duration = (charging_energy / (cs.fastchargingpower * pev.charging_effi)) * 60

            cs_charging_cost = charging_energy * chargingprice + ept_charging_duration/60 * UNITtimecost

            athome_remainE = (pev.maxBCAPA * pev.req_SOC - rear_consump_energy)
            athome_soc = athome_remainE / pev.maxBCAPA
            home_charging_cost = (pev.maxBCAPA * pev.final_soc - athome_remainE) * cs.TOU_price[int(cur_time / 60)]

            ept_WT, at_list, diff = cs.get_ept_WT(pev,ept_arrtime,  ept_charging_duration, cur_time, graph)
            # ept_WT, at_list, diff = cs.get_current_WT(pev, cur_time, graph)

            weight = ept_WT
            info.append((cs, weight, ept_driving_cost, front_path, rear_path, front_path_distance, rear_path_distance,
                         front_d_time, rear_d_time, fpath_weight, rpath_weight, ept_WT,
                         ept_charging_duration, cs_charging_cost, home_charging_cost, ept_arrtime))

            # print('actions: ',cs.id, req_soc)

    return info

def get_greedy_shortwt_fleet_reserv(request_be_EV, CS_list, graph):
    for pev in request_be_EV:
        # charging_energy = pev.maxBCAPA * (pev.req_SOC - pev.curr_SOC)
        # charging_duration = (charging_energy / (60 * pev.charging_effi))
        # pev.ept_charging_duration = charging_duration * 60

        # print('=====================================================================')
        # print('be ID:{0:3} S:{1:3} D:{2:3} CurSOC:{3:0.2f} ReqSOC:{4:0.2f} Tstart:{5:0.2f} Tarr:{6:0.2f}'
        #       .format(pev.id, pev.source, pev.destination, pev.curr_SOC, pev.req_SOC, pev.t_start, pev.ept_arrtime))
        candi = []
        candi = get_feature_state_shortwt_fleet_reserv(pev.t_start, pev, CS_list, graph, 0)

        # for cs, _, _, _, _, _, _, _, _, _, _, eptWT, ept_charduration, _, _, ept_arrtime in candi:
        #     print('ID: {0:3}  eptWT: {1:.2f}   eptCharduration: {3:.2f}   Len_reserv: {2:.2f}'.format(cs.id, eptWT, len(
        #         cs.reserve_ev), ept_charduration))

        candi.sort(key=lambda e: e[1])

        (cs, weight, ept_driving_cost, front_path, rear_path, front_path_distance, rear_path_distance, ept_front_d_time,
         ept_rear_d_time, fpath_weight, rpath_weight, ept_WT, ept_charduration, ept_cs_charging_cost,
         ept_home_charging_cost,
         ept_arrtime) = candi[0]

        pev.front_path = front_path
        pev.rear_path = rear_path
        pev.path = front_path + rear_path[1:]
        pev.ept_arrtime = ept_arrtime
        pev.true_arrtime = get_true_arrtime(pev, cs, graph)
        pev.ept_waitingtime = ept_WT
        pev.ept_charging_duration = ept_charduration
        pev.ept_totaldrivingtime = ept_front_d_time + ept_rear_d_time
        pev.cs = cs
        pev.cschargingprice = cs.price[int(pev.cschargingstarttime / 5)]
        pev.eptcschargingstarttime = pev.ept_arrtime + ept_WT
        cs.recieve_request(pev)

    for cs in CS_list:
        cs.sim_finish(graph)
        # print(cs.id, end=', ')
    # print('\n=====================================================================')

    tot_wt = 0
    tot_cost = 0
    for pev in request_be_EV:
        # print(
        #     'result,  ID: {0:3},  CSID: {1:3},  CurSOC: {2:.2f},  ReqSOC: {3:.2f},  Tstart: {4:5.2f},  EptTarr: {5:5.2f},  TruTarr: {10:5.2f},  diffTarr: {11:5.2f},  WT: {6:5.2f},  eptWT: {8:5.2f},  diffWT: {9:5.2f},  ChaStart: {7:5.2f},  finTime: {12:5.2f}'
        #     .format(pev.id, pev.cs.id, pev.curr_SOC, pev.req_SOC, pev.t_start, pev.ept_arrtime, pev.true_waitingtime,
        #             pev.cschargingstarttime, pev.ept_waitingtime, pev.time_diff_WT, pev.true_arrtime,
        #             pev.true_arrtime - pev.ept_arrtime, pev.curr_time))
        tot_wt += pev.true_waitingtime
        tot_cost += pev.totalcost



def sim_result_general_presentation_last(nth, graph, resultdir, numev, **results):
    print('makeing figures...')
    keyname = ''
    for key in results.keys():
        keyname += '_'+ key
    basepath = os.getcwd()
    resultdir = resultdir+'/result{}_{}'.format(keyname, nth)
    print(os.path.join(basepath, resultdir))
    dirpath = os.path.join(basepath, resultdir)
    createFolder(dirpath)

    keylist = list(results.keys())
    linetype = ['-', '--', ':', '-.']

    fig = plt.figure(figsize=(12, 12), dpi=300)
    for i in range(numev):
        kth_result=1
        cnt = 0
        totcost = []
        for key, EVlist in results.items():
            pev = EVlist[i]
            xx = []
            yy = []
            nth=0
            ax = fig.add_subplot(2, 2, cnt+1)
            ax.set_title(key)
            for nid in pev.path:
                x, y = graph.nodes_xy(nid)
                plt.text(x,y,str(nth))
                xx.append(x)
                yy.append(y)
                nth+=1
            ax.plot(xx, yy, linetype[cnt])
            cnt += 1
            if pev.cs != None:
                cs_x, cs_y = graph.nodes_xy(pev.cs.id)
                ax.plot(cs_x, cs_y, 'D', label=key+' EVCS')
            kth_result+=1

            s_x, s_y = graph.nodes_xy(pev.source)
            # ax.set_xlim(graph.minx - 1, graph.maxx + 1)
            # ax.set_ylim(graph.miny - 1, graph.maxy + 1)
            ax.plot(s_x, s_y, 'p', label='Source')

            d_x, d_y = graph.nodes_xy(pev.destination)
            ax.set_xlim(graph.minx-1, graph.maxx+1)
            ax.set_ylim(graph.miny-1, graph.maxy+1)
            ax.plot(d_x, d_y, 'p', label='Destination')
            plt.legend()

            totcost.append(pev.totalcost)

        ax = fig.add_subplot(2, 2, 4)
        ax.set_title('total cost')
        ax.bar(keylist, totcost)

        fig = plt.gcf()
        fig.savefig('{}/route_{}.png'.format(resultdir, i), facecolor='#eeeeee', dpi=300)
        plt.clf()


    plt.title('Selected EVCS')
    plt.xlabel('EV index')
    plt.ylabel('EVCS ID')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.cs.id)
        plt.plot(range(len(r1_list)), r1_list,'x',  label=key)
    plt.legend()
    # plt.xlim(graph.minx, graph.maxx)
    # plt.ylim(graph.miny, graph.maxy)
    fig = plt.gcf()
    fig.savefig('{}/EVCS.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()


    plt.figure(figsize=(12, 6), dpi=300)

    plt.title('CS Charging Cost')
    plt.xlabel('EV ID')
    plt.ylabel('Cost($)')
    cnt=0
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.cschargingcost)
        plt.plot(r1_list, linetype[cnt], label=key)
        cnt+=1
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/CS Charging Cost.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()

    plt.title('Home Charging Cost')
    plt.xlabel('EV ID')
    plt.ylabel('Cost($)')
    cnt=0
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.homechargingcost)
        plt.plot(r1_list, linetype[cnt], label=key)
        cnt+=1
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/Home Charging Cost.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()


    plt.title('Driving distance')
    plt.xlabel('EV ID')
    plt.ylabel('Distance(km)')
    cnt = 0
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.totaldrivingdistance)
        plt.plot(r1_list, linetype[cnt], label=key)
        cnt += 1
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/Driving distance.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()

    plt.title('Distance from S to EVCS')
    plt.xlabel('EV ID')
    plt.ylabel('Distance(km)')
    cnt = 0
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.fdist)
        plt.plot(r1_list, linetype[cnt], label=key)
        cnt += 1
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/Distance from S to EVCS.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()

    plt.title('Driving time')
    plt.xlabel('EV ID')
    plt.ylabel('Time(h)')
    cnt = 0
    for key, EVlist in results.items():
        numev = len(EVlist)
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.totaldrivingtime)
        plt.plot(r1_list, linetype[cnt], label=key)
        cnt += 1
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/Driving time.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()

    plt.title('CS Charging energy')
    plt.xlabel('EV ID')
    plt.ylabel('Energy(kWh)')
    cnt = 0
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.cscharingenergy)
        plt.plot(r1_list, linetype[cnt], label=key)
        cnt += 1
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/CS Charging energy.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()

    plt.title('Home Charging energy')
    plt.xlabel('EV ID')
    plt.ylabel('Energy(kWh)')
    cnt = 0
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.homechargingenergy)
        plt.plot(r1_list, linetype[cnt], label=key)
        cnt += 1
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/Home Charging energy.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()

    plt.title('CS Charging time')
    plt.xlabel('EV ID')
    plt.ylabel('Time(h)')
    cnt = 0
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.cschargingtime)
        plt.plot(r1_list, linetype[cnt], label=key)
        cnt += 1
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/CS Charging time.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()

    plt.title('Waiting time')
    plt.xlabel('EV ID')
    plt.ylabel('Time(h)')
    cnt = 0
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.cschargingwaitingtime)
        plt.plot(r1_list, linetype[cnt], label=key)
        cnt += 1
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/Waiting time.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()

    plt.title('Total travel time')
    plt.xlabel('EV ID')
    plt.ylabel('Time(h)')
    cnt = 0
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.cschargingwaitingtime + ev.cschargingtime + ev.totaldrivingtime)
        plt.plot(r1_list, linetype[cnt], label=key)
        cnt += 1
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/Total travel time.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()

    plt.title('Total travel cost')
    plt.xlabel('EV ID')
    plt.ylabel('Cost($)')
    cnt = 0
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.totalcost)
        plt.plot(r1_list, linetype[cnt], label=key)
        cnt += 1
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/Total travel cost.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()

    plt.title('EV SOC')
    plt.xlabel('EV ID')
    plt.ylabel('SOC(%)')
    cnt = 0
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.init_SOC)
        plt.plot(r1_list, linetype[cnt], label=key)
        cnt += 1
    cnt = 0
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.curr_SOC)
        plt.plot(r1_list, linetype[cnt], label=key)
        cnt += 1
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/ev SOC.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()

    #=======================================================================================

def sim_result_text_last(nth, CS_list, graph, resultdir, **results):
    print('makeing documents...')
    keyname = ''
    for key in results.keys():
        keyname += key + '_'
    print(keyname)

    fw = open('{}/data_{}_{}.txt'.format(resultdir, keyname, nth), 'w', encoding='UTF8')

    fw.write('\ncs price\n')
    for cs in CS_list:
        r1_list = []
        for p in cs.price:
            r1_list.append(p)
        fw.write(str(cs.id) + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')

    fw.write('\ncs wait\n')
    for cs in CS_list:
        r1_list = []
        for p in cs.waittime:
            r1_list.append(p)
        fw.write(str(cs.id) + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')


    fw.write('\nlink velocity\n')
    for l_id in graph.link_data.keys():
        fw.write(str(l_id) + '\t' + str(graph.traffic_info[l_id]) + '\n')

    fw.close()



    fw = open('{}/result_{}_{}.txt'.format(resultdir, keyname, nth), 'w', encoding='UTF8')


    fw.write('\nSum total travel cost\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.totalcost)
        fw.write(key + '\t' + str(sum(r1_list)) + '\n')

    fw.write('\nSum total expense_time_part\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.expense_time_part)
        fw.write(key + '\t' + str(sum(r1_list)) + '\n')

    fw.write('\nSum total expense_cost_part\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.expense_cost_part)
        fw.write(key + '\t' + str(sum(r1_list)) + '\n')


    fw.write('\nSum total travel time\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.cschargingwaitingtime + ev.cschargingtime + ev.totaldrivingtime)
        fw.write(key + '\t' + str(sum(r1_list)) + '\n')

    fw.write('\nSum total driving time\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.totaldrivingtime)
        fw.write(key + '\t' + str(sum(r1_list)) + '\n')
    fw.write('\nSum total cs charging time\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.cschargingtime)
        fw.write(key + '\t' + str(sum(r1_list)) + '\n')

    fw.write('\nSum total cs waiting time\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.cschargingwaitingtime)
        fw.write(key + '\t' + str(sum(r1_list)) + '\n')




    fw.write('\nSum total driving distance\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.totaldrivingdistance)
        fw.write(key + '\t' + str(sum(r1_list)) + '\n')



    fw.write('\nSum total cs chargingcost\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.cschargingcost)
        fw.write(key + '\t' + str(sum(r1_list)) + '\n')


    fw.write('\nSum total home chargingcost\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.homechargingcost)
        fw.write(key + '\t' + str(sum(r1_list)) + '\n')


    fw.write('\ncs\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.cs.id)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')

    fw.write('cs req soc\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.req_SOC)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')




    fw.write('\nTotal travel cost\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.totalcost)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')


    fw.write('\nTotal travel expense_cost_part\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.expense_cost_part)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')

    fw.write('\nTotal travel expense_time_part\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.expense_time_part)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')




    fw.write('\ncs driving distance\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.csdistance)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')

    fw.write('\ncs driving time\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.csdrivingtime)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')

    fw.write('\ncs charging energy\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.cscharingenergy)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')

    fw.write('\ncs charging cost\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.cschargingcost)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')

    fw.write('\ncs waiting time\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.cschargingwaitingtime)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')

    fw.write('\ncs charging time\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.cschargingtime)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')







    fw.write('\nhome driving distance\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.homedrivingdistance)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')

    fw.write('\nhome driving time\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.homedrivingtime)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')

    fw.write('\nhome charging energy\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.homechargingenergy)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')

    fw.write('\nhome charging cost\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.homechargingcost)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')

    fw.write('\nhome charging time\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.homechargingtime)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')




    fw.write('\nTotal Driving distance\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.totaldrivingdistance)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')

    fw.write('\nTotal Driving time\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.totaldrivingtime)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')

    fw.write('\nTotal travel time\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.cschargingwaitingtime + ev.cschargingtime + ev.totaldrivingtime)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')



    fw.write('\nSource\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.source)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')


    fw.write('\ncharging price\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.cschargingprice)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')



    fw.write('\ninit_SOC\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.init_SOC)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')

    fw.write('at cs soc\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.cssoc)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')



    fw.write('at home soc\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.homesoc)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')

    fw.write('\nt_start\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.t_start)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')


    fw.write('\nCharging start time\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.cschargingstarttime)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')











    fw.write('\nPath\n')
    for key, EVlist in results.items():
        fw.write(key+'\n')
        for ev in EVlist:
            for value in ev.path:
                fw.write(str(value) + '\t')
            fw.write('\n')
        fw.write('\n')


    fw.close()


def sim_result_general_presentation_fleet(nth, CS_list, graph, resultdir, **results):
    print('makeing figures...')
    keyname = ''
    for key in results.keys():
        keyname += '_' + key
    basepath = os.getcwd()
    resultdir = resultdir + '/result{}_{}'.format(keyname, nth)
    print(os.path.join(basepath, resultdir))
    dirpath = os.path.join(basepath, resultdir)
    createFolder(dirpath)

    keylist = list(results.keys())
    linetype = ['-.', '-.', '-.', '-.']







    fig = plt.figure(figsize=(12, 12), dpi=600)
    for i in range(90,100):
        kth_result = 1
        cnt = 0
        totcost = []
        for key, EVlist in results.items():
            pev = EVlist[i]
            xx = []
            yy = []
            nth = 0
            ax = fig.add_subplot(2, 2, cnt + 1)
            ax.set_title(key)

            for lpair in graph.link_pair_data:
                xxx, yyy = [], []
                fnode, tnode = lpair
                lid = graph.link_pair_data[fnode, tnode]
                x, y = graph.nodes_xy(fnode)
                xxx.append(x)
                yyy.append(y)
                x, y = graph.nodes_xy(tnode)
                # plt.text(x, y, str(nth))
                xxx.append(x)
                yyy.append(y)
                lid = lid[0]
                maxspd = graph.link_data[lid]['MAX_SPD']

                if maxspd == 120:
                    ax.plot(xxx, yyy, '-', color='green')
                elif maxspd == 80:
                    ax.plot(xxx, yyy, '-', color='orange')
                else:
                    ax.plot(xxx, yyy, '-', color='red')

            for cs in graph.cs_info.keys():
                x, y = graph.nodes_xy(cs)
                ax.plot(x, y, '*', color='gray')



            for nid in pev.path:
                x, y = graph.nodes_xy(nid)

                xx.append(x)
                yy.append(y)
                nth += 1
            ax.plot(xx, yy, '-.', linewidth=3)
            cnt += 1
            if pev.cs != None:
                cs_x, cs_y = graph.nodes_xy(pev.cs.id)
                ax.plot(cs_x, cs_y, 'D', label='EVCS')
                plt.text(cs_x+0.5, cs_y, 'EVCS')
                plt.text(cs_x+0.5, cs_y-1, 'WT:{0:4.2f}'.format(pev.true_waitingtime))
                plt.text(cs_x+0.5, cs_y-2, 'CT:{0:4.2f}'.format(pev.true_charging_duration))
            kth_result += 1

            ax.set_xlim(graph.minx - 1, graph.maxx + 1)
            ax.set_ylim(graph.miny - 1, graph.maxy + 1)


            s_x, s_y = graph.nodes_xy(pev.source)
            ax.plot(s_x, s_y, 'p', label='Source')
            plt.text(s_x+0.5, s_y, 'S')

            d_x, d_y = graph.nodes_xy(pev.destination)
            ax.plot(d_x, d_y, 'p', label='Destination')
            plt.text(d_x+0.5, d_y, 'D')
            plt.text(d_x+0.5, d_y-1, 'DT:{0:4.2f}'.format(pev.totaldrivingtime))


            totcost.append(pev.totalcost)

        # ax = fig.add_subplot(2, 2, 4)
        # ax.set_title('total cost')
        # ax.bar(keylist, totcost)
        # plt.legend()
        fig = plt.gcf()
        fig.savefig('{}/route_{}.png'.format(resultdir, i), facecolor='white', dpi=600)
        plt.clf()

    # plt.title('Selected EVCS')
    # plt.xlabel('EV index')
    # plt.ylabel('EVCS ID')
    # for key, EVlist in results.items():
    #     r1_list = []
    #     for ev in EVlist:
    #         r1_list.append(ev.cs.id)
    #     plt.plot(range(len(r1_list)), r1_list, 'x', label=key)
    # plt.legend()
    # # plt.xlim(graph.minx, graph.maxx)
    # # plt.ylim(graph.miny, graph.maxy)
    # fig = plt.gcf()
    # fig.savefig('{}/EVCS.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    # plt.clf()




    #=======================================================================================

def sim_result_text_fleet(nth, CS_list, graph, resultdir, **results):
    print('makeing documents...')
    keyname = ''
    for key in results.keys():
        keyname += key + '_'
    print(keyname)

    fw = open('{}/data_{}_{}.txt'.format(resultdir, keyname, nth), 'w', encoding='UTF8')

    fw.write('\nunknown EV charging\n')
    for cs in CS_list:
        fw.write(str(cs.id) + '\t')

        for uev, arrtime in cs.unknown_ev:
            fw.write(str(uev.id) + ', ' + str(uev.true_arrtime) + '\t')
        fw.write('\n')


    fw.write('\ncs price\n')
    for cs in CS_list:
        r1_list = []
        for p in cs.price:
            r1_list.append(p)
        fw.write(str(cs.id) + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')

    fw.write('\ncs wait\n')
    for cs in CS_list:
        r1_list = []
        for p in cs.waittime:
            r1_list.append(p)
        fw.write(str(cs.id) + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')


    fw.write('\nlink velocity\n')
    for l_id in graph.link_data.keys():
        fw.write(str(l_id) + '\t' + str(graph.traffic_info[l_id]) + '\n')

    fw.close()



    test = []
    for key, (EVlist, _) in results.items():
        rlist = []
        for ev in EVlist:
            rlist.append(ev.true_waitingtime + ev.true_charging_duration + ev.totaldrivingtime)
        test.append(sum(rlist)/len(EVlist))



    fw_total = open('{}/summary.txt'.format(resultdir), 'a', encoding='UTF8')
    # fw_total.write('total tarvel time'+'\t\t\t\t')
    # fw_total.write('total driving time'+'\t\t\t\t')
    # fw_total.write('total charging time'+'\t\t\t\t')
    # fw_total.write('total waiting time'+'\t\t\t\t')
    # fw_total.write('total total driving distance'+'\t\t\t\t')
    # fw_total.write('total total front distance'+'\t\t\t\t')
    # fw_total.write('total total rear distance'+'\t\t\t\t')
    # fw_total.write('total total charging energy'+'\n')

    # if test[3] > test[0] and test[3] > test[1] and test[1] > test[0]:


    # fw_total.write(str(test[3]-test[0]) + '\t')
    # fw_total.write(str(test[3] - test[1]) + '\t')



    fw = open('{}/result_{}_{}.txt'.format(resultdir, keyname, nth), 'w', encoding='UTF8')

    fw.write('\navg total travel time\n')
    for key, (EVlist,_) in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.true_waitingtime + ev.true_charging_duration + ev.totaldrivingtime)
        # fw.write(key + '\t' + str(sum(r1_list)) + '\t' + str(sum(r1_list)/len(EVlist)) +  '\n')
        fw.write(key + '\t' + str(sum(r1_list)/len(EVlist)) +  '\n')

        fw_total.write(str(sum(r1_list)/len(EVlist)) +  '\t')


    fw.write('\navg total driving time\n')
    for key, (EVlist,_) in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.totaldrivingtime)
        # fw.write(key + '\t' + str(sum(r1_list)) + '\t' + str(sum(r1_list)/len(EVlist)) +  '\n')
        fw.write(key + '\t' + str(sum(r1_list)/len(EVlist)) +  '\n')
        fw_total.write(str(sum(r1_list) / len(EVlist)) + '\t')




    fw.write('\navg total cs charging time\n')
    for key, (EVlist,_) in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.true_charging_duration)
        # fw.write(key + '\t' + str(sum(r1_list)) + '\t' + str(sum(r1_list)/len(EVlist)) +  '\n')
        fw.write(key + '\t' + str(sum(r1_list)/len(EVlist)) +  '\n')
        fw_total.write(str(sum(r1_list) / len(EVlist)) + '\t')

    fw.write('\navg total cs waiting time\n')
    for key, (EVlist,_) in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.true_waitingtime)
        # fw.write(key + '\t' + str(sum(r1_list)) + '\t' + str(sum(r1_list)/len(EVlist)) +  '\n')
        fw.write(key + '\t' + str(sum(r1_list)/len(EVlist)) +  '\n')
        fw_total.write(str(sum(r1_list) / len(EVlist)) + '\t')

    fw.write('\navg total driving distance\n')
    for key, (EVlist,_) in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.totaldrivingdistance)
        # fw.write(key + '\t' + str(sum(r1_list)) + '\t' + str(sum(r1_list)/len(EVlist)) +  '\n')
        fw.write(key + '\t' + str(sum(r1_list)/len(EVlist)) +  '\n')
        fw_total.write(str(sum(r1_list) / len(EVlist)) + '\t')

    fw.write('\navg total front driving distance\n')
    for key, (EVlist,_) in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.fdist)
        # fw.write(key + '\t' + str(sum(r1_list)) + '\t' + str(sum(r1_list)/len(EVlist)) +  '\n')
        fw.write(key + '\t' + str(sum(r1_list)/len(EVlist)) +  '\n')
        fw_total.write(str(sum(r1_list) / len(EVlist)) + '\t')

    fw.write('\navg total rear driving distance\n')
    for key, (EVlist,_) in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.rdist)
        # fw.write(key + '\t' + str(sum(r1_list)) + '\t' + str(sum(r1_list)/len(EVlist)) +  '\n')
        fw.write(key + '\t' + str(sum(r1_list)/len(EVlist)) +  '\n')
        fw_total.write(str(sum(r1_list) / len(EVlist)) + '\t')



    fw.write('\navg total charging energy\n')
    for key, (EVlist, _) in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.cscharingenergy)
        # fw.write(key + '\t' + str(sum(r1_list)) + '\t' + str(sum(r1_list)/len(EVlist)) +  '\n')
        fw.write(key + '\t' + str(sum(r1_list) / len(EVlist)) + '\n')
        fw_total.write(str(sum(r1_list) / len(EVlist)) + '\t')
    fw_total.write('\n')

    fw.write('\nCS distribution\n')
    for key, (EVlist, CS_list) in results.items():
        fw.write('{}\t'.format(key))
        for cs in CS_list:
            num_ev = 0
            for cp in cs.cplist:
                num_ev += len(cp.charging_ev)
            fw.write('{}\t'.format(num_ev))
        fw.write('\n')




    fw.write('\nSum ept total travel time\n')
    for key, (EVlist,_) in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.ept_waitingtime + ev.ept_charging_duration + ev.ept_totaldrivingtime)
        fw.write(key + '\t' + str(sum(r1_list)) + '\t' + str(sum(r1_list)/len(EVlist)) +  '\n')

    fw.write('\nSum ept total driving time\n')
    for key, (EVlist,_) in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.ept_totaldrivingtime)
        fw.write(key + '\t' + str(sum(r1_list)) + '\t' + str(sum(r1_list)/len(EVlist)) +  '\n')

    fw.write('\nSum ept total cs charging time\n')
    for key, (EVlist,_) in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.ept_charging_duration)
        fw.write(key + '\t' + str(sum(r1_list)) + '\t' + str(sum(r1_list)/len(EVlist)) +  '\n')


    fw.write('\nSum ept total cs waiting time\n')
    for key, (EVlist,_) in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.ept_waitingtime)
        fw.write(key + '\t' + str(sum(r1_list)) + '\t' + str(sum(r1_list)/len(EVlist)) +  '\n')

    fw.write('\ncs\n')
    for key, (EVlist,_) in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.cs.id)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')

    fw.write('cs req soc\n')
    for key, (EVlist,_) in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.req_SOC)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')


    fw.write('\ncs charging cost\n')
    for key, (EVlist,_) in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.cschargingcost)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')

    fw.write('\ncs waiting time\n')
    for key, (EVlist,_) in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.true_waitingtime)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')

    fw.write('\ncs charging time\n')
    for key, (EVlist,_) in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.true_charging_duration)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')



    fw.write('\nTotal Driving distance\n')
    for key, (EVlist,_) in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.totaldrivingdistance)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')

    fw.write('\nTotal Driving time\n')
    for key, (EVlist,_)in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.totaldrivingtime)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')

    fw.write('\nTotal travel time\n')
    for key, (EVlist,_) in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.true_waitingtime + ev.true_charging_duration + ev.totaldrivingtime)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')



    fw.write('\nSource\n')
    for key, (EVlist,_) in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.source)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')


    fw.write('\ncharging price\n')
    for key, (EVlist,_) in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.cschargingprice)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')



    fw.write('\ninit_SOC\n')
    for key, (EVlist,_) in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.init_SOC)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')

    fw.write('at cs soc\n')
    for key, (EVlist,_) in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.cssoc)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')



    fw.write('at home soc\n')
    for key, (EVlist,_) in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.homesoc)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')

    fw.write('\nt_start\n')
    for key, (EVlist,_) in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.t_start)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')


    fw.write('\nCharging start time\n')
    for key, (EVlist,_) in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.cschargingstarttime)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')






    fw.write('\nPath\n')
    for key, (EVlist,_) in results.items():
        fw.write(key+'\n')
        for ev in EVlist:
            for value in ev.path:
                fw.write(str(value) + '\t')
            fw.write('\n')
        fw.write('\n')


    fw.close()

    fw_total.close()