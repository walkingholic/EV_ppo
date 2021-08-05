import data_gen
import pprint as pp
import numpy as np
import matplotlib.pyplot as plt
import heapq
import datetime
import os
import setting as st

# seed = st.random_seed
# np.random.seed(seed)




class Graph_simple:
    def __init__(self):
        # self.link_data, self.node_data, self.traffic_info, self.cs_info, self.minx, self.miny, self.maxx, self.maxy = data_gen.network_info_simple()
        self.link_data, self.node_data, self.traffic_info, self.cs_info, self.minx, self.miny, self.maxx, self.maxy = data_gen.network_info_simple_25()
        self.num_node = len(self.node_data)
        self.num_link = len(self.link_data)
        self.neighbors_list = {}
        self.link_pair_data = {}
        self.source_node_set = set()
        self.destination_node_set = set()

        # print('link 4070105400', self.link_data[4070105400])

        print('Num links from link data', len(self.link_data.keys()))
        print('Num node from link data', len(self.node_data.keys()))
        count = 0
        errorlink = 0

        for l in self.link_data.keys():
            maxspd = self.link_data[l]['MAX_SPD']
            self.traffic_info[l] = list(np.random.random_integers(maxspd - maxspd * 0.3, maxspd, 288))
            # self.traffic_info[l] = list(np.ones(288)*maxspd)

        print('Num links from traffic data', len(self.traffic_info.keys()))
        print('Modified Num link', len(self.link_data.keys()))

        for l in self.link_data.keys():
            self.source_node_set.add(self.link_data[l]['F_NODE'])
            self.destination_node_set.add(self.link_data[l]['T_NODE'])

        # print(len(self.node_data), len(self.source_node_set), len(self.destination_node_set))

        for lid in self.link_data.keys():

            if self.link_data[lid]['F_NODE'] in self.neighbors_list:
                self.neighbors_list[self.link_data[lid]['F_NODE']].append(self.link_data[lid]['T_NODE'])
            else:
                self.neighbors_list[self.link_data[lid]['F_NODE']] = [self.link_data[lid]['T_NODE']]

            if (self.link_data[lid]['F_NODE'], self.link_data[lid]['T_NODE']) in self.link_pair_data:
                self.link_pair_data[(self.link_data[lid]['F_NODE']), self.link_data[lid]['T_NODE']].append(lid)
            else:
                self.link_pair_data[(self.link_data[lid]['F_NODE']), self.link_data[lid]['T_NODE']] = [lid]

    def reset_traffic_info(self):
        np.random.seed(seed)
        for l in self.link_data.keys():
            # print('before', self.traffic_info[l])
            maxspd = self.link_data[l]['MAX_SPD']
            self.traffic_info[l] = list(np.random.random_integers(maxspd - maxspd * 0.3, maxspd, 288))
            # print('after', self.traffic_info[l])


    def get_link_id(self, fnode, tnode):
        return len(self.link_pair_data[(fnode,tnode)]), self.link_pair_data[(fnode,tnode)]

    def get_node_id(self, idx):
        # print(self.node_data[idx])
        return self.node_data[idx]['NODE_ID']

    def neighbors(self, fnode):                 #  return
        # pp.pprint(self.neighbors_list)
        return self.neighbors_list[fnode]

    def weight(self, link_id):
        # link_id =self.get_link_id(fromnode, tonode)
        return self.link_data[link_id]['WEIGHT']

    def weight(self, fromnode, tonode):
        n, link_id_list = self.get_link_id(fromnode, tonode)
        lid = link_id_list[0]
        return self.link_data[link_id_list[0]]['WEIGHT']

    def distance(self, link_id):
        # link_id = self.get_link_id(fromnode, tonode)
        return self.link_data[link_id]['LENGTH']

    def distance(self, fromnode, tonode):
        # link_id = self.get_link_id(fromnode, tonode)
        n, link_id_list = self.get_link_id(fromnode, tonode)

        return self.link_data[link_id_list[0]]['LENGTH']

    def velocity(self, link_id):
        # link_id = self.get_link_id(fromnode, tonode)
        return self.link_data[link_id]['CUR_SPD']

    def velocity(self, fromnode, tonode, tidx):
        # link_id = self.get_link_id(fromnode, tonode)
        n, link_id_list = self.get_link_id(fromnode, tonode)
        link_id = link_id_list[0]
        return self.traffic_info[link_id][tidx]

    def nodes_xy(self, nidx):
        return (self.node_data[nidx]['long'], self.node_data[nidx]['lat'])

    def get_path_distance(self, path):
        distance=0
        for i in range(len(path)-1):
            fromenode = path[i]
            tonode = path[i+1]
            distance = distance+self.distance(fromenode, tonode)
        return distance

    def get_path_weight(self, path):
        weight = 0
        for i in range(len(path) - 1):
            fromenode = path[i]
            tonode = path[i + 1]
            weight = weight + self.weight(fromenode, tonode)
        return weight

    def get_path_avg_velo(self, path, tidx):
        sumvelo = 0
        for i in range(len(path) - 1):
            fromenode = path[i]
            tonode = path[i + 1]
            sumvelo = sumvelo + self.velocity(fromenode, tonode, tidx)
        return sumvelo/(len(path) - 1)

    def get_path_drivingtime(self, path, tidx):
        dtime = 0.0

        for i in range(len(path) - 1):
            fromenode = path[i]
            tonode = path[i + 1]
            dtime = dtime + self.distance(fromenode, tonode) / self.velocity(fromenode, tonode, tidx)

        return dtime

class Graph_simple_100:
    def __init__(self):
        self.link_data, self.node_data, self.traffic_info, self.cs_info, self.minx, self.miny, self.maxx, self.maxy = data_gen.network_info_simple_100()
        # self.link_data, self.node_data, self.traffic_info, self.cs_info, self.minx, self.miny, self.maxx, self.maxy = data_gen.network_info_simple_25()
        self.num_node =  len(self.node_data)
        self.num_link = len(self.link_data)
        self.neighbors_list = {}
        self.link_pair_data = {}
        self.source_node_set = set()
        self.destination_node_set = set()

        # print('link 4070105400', self.link_data[4070105400])

        print('Num links from link data', len(self.link_data.keys()))
        print('Num node from link data', len(self.node_data.keys()))
        count = 0
        errorlink = 0

        for l in self.link_data.keys():

            if l in self.traffic_info.keys():
                count += 1
                if len(self.traffic_info[l]) < 288:
                    avg = sum(self.traffic_info[l]) / len(self.traffic_info[l])
                    for i in range(288 - len(self.traffic_info[l])):
                        self.traffic_info[l].append(int(avg))

                    errorlink += 1
            else:
                maxspd = self.link_data[l]['MAX_SPD']
                self.traffic_info[l] = list(np.random.random_integers(maxspd-maxspd*0.3, maxspd, 288))




        print('Num links from traffic data', len(self.traffic_info.keys()))
        print('Modified Num link', len(self.link_data.keys()))

        for l in self.link_data.keys():
            self.source_node_set.add(self.link_data[l]['F_NODE'])
            self.destination_node_set.add(self.link_data[l]['T_NODE'])

        # print(len(self.node_data), len(self.source_node_set), len(self.destination_node_set))

        for lid in self.link_data.keys():

            if self.link_data[lid]['F_NODE'] in self.neighbors_list:
                self.neighbors_list[self.link_data[lid]['F_NODE']].append(self.link_data[lid]['T_NODE'])
            else:
                self.neighbors_list[self.link_data[lid]['F_NODE']] = [self.link_data[lid]['T_NODE']]

            if (self.link_data[lid]['F_NODE'], self.link_data[lid]['T_NODE']) in self.link_pair_data:
                self.link_pair_data[(self.link_data[lid]['F_NODE']), self.link_data[lid]['T_NODE']].append(lid)
            else:
                self.link_pair_data[(self.link_data[lid]['F_NODE']), self.link_data[lid]['T_NODE']] = [lid]

    def reset_traffic_info(self):

        for l in self.link_data.keys():
            # print('before', self.traffic_info[l])
            maxspd = self.link_data[l]['MAX_SPD']
            self.traffic_info[l] = list(np.random.random_integers(maxspd-maxspd*0.3, maxspd, 288))
            # print('after', self.traffic_info[l])

    def get_link_id(self, fnode, tnode):
        return len(self.link_pair_data[(fnode,tnode)]), self.link_pair_data[(fnode,tnode)]

    def get_node_id(self, idx):
        # print(self.node_data[idx])
        return self.node_data[idx]['NODE_ID']

    def neighbors(self, fnode):                 #  return
        # pp.pprint(self.neighbors_list)
        return self.neighbors_list[fnode]

    def weight(self, link_id):
        # link_id =self.get_link_id(fromnode, tonode)
        return self.link_data[link_id]['WEIGHT']

    def weight(self, fromnode, tonode):
        n, link_id_list = self.get_link_id(fromnode, tonode)
        lid = link_id_list[0]
        return self.link_data[link_id_list[0]]['WEIGHT']

    def distance(self, link_id):
        # link_id = self.get_link_id(fromnode, tonode)
        return self.link_data[link_id]['LENGTH']

    def distance(self, fromnode, tonode):
        # link_id = self.get_link_id(fromnode, tonode)
        n, link_id_list = self.get_link_id(fromnode, tonode)

        return self.link_data[link_id_list[0]]['LENGTH']

    def velocity(self, link_id):
        # link_id = self.get_link_id(fromnode, tonode)
        return self.link_data[link_id]['CUR_SPD']

    def velocity(self, fromnode, tonode, tidx):
        # link_id = self.get_link_id(fromnode, tonode)
        n, link_id_list = self.get_link_id(fromnode, tonode)
        link_id = link_id_list[0]
        return self.traffic_info[link_id][tidx]

    def nodes_xy(self, nidx):
        return (self.node_data[nidx]['long'], self.node_data[nidx]['lat'])

    def get_path_distance(self, path):
        distance=0
        for i in range(len(path)-1):
            fromenode = path[i]
            tonode = path[i+1]
            distance = distance+self.distance(fromenode, tonode)
        return distance

    def get_path_weight(self, path):
        weight = 0
        for i in range(len(path) - 1):
            fromenode = path[i]
            tonode = path[i + 1]
            weight = weight + self.weight(fromenode, tonode)
        return weight

    def get_path_avg_velo(self, path, tidx):
        sumvelo = 0
        for i in range(len(path) - 1):
            fromenode = path[i]
            tonode = path[i + 1]
            sumvelo = sumvelo + self.velocity(fromenode, tonode, tidx)
        return sumvelo/(len(path) - 1)

    def get_path_drivingtime(self, path, tidx):
        dtime = 0.0

        for i in range(len(path) - 1):
            fromenode = path[i]
            tonode = path[i + 1]
            dtime = dtime + self.distance(fromenode, tonode) / self.velocity(fromenode, tonode, tidx)

        return dtime

class Graph_simple_39:
    def __init__(self):
        self.link_data, self.node_data, self.traffic_info, self.cs_info, self.minx, self.miny, self.maxx, self.maxy = data_gen.network_info_simple_39()
        # self.link_data, self.node_data, self.traffic_info, self.cs_info, self.minx, self.miny, self.maxx, self.maxy = data_gen.network_info_simple_25()
        self.num_node =  len(self.node_data)
        self.num_link = len(self.link_data)
        self.neighbors_list = {}
        self.link_pair_data = {}
        self.source_node_set = set()
        self.destination_node_set = set()

        # print('link 4070105400', self.link_data[4070105400])

        print('Num links from link data', len(self.link_data.keys()))
        print('Num node from link data', len(self.node_data.keys()))
        count = 0
        errorlink = 0

        for l in self.link_data.keys():
            maxspd = self.link_data[l]['MAX_SPD']
            self.traffic_info[l] = list(np.random.random_integers(maxspd-maxspd*0.3, maxspd, 288))
            # self.traffic_info[l] = list(np.ones(288)*maxspd)




        print('Num links from traffic data', len(self.traffic_info.keys()))
        print('Modified Num link', len(self.link_data.keys()))

        for l in self.link_data.keys():
            self.source_node_set.add(self.link_data[l]['F_NODE'])
            self.destination_node_set.add(self.link_data[l]['T_NODE'])

        # print(len(self.node_data), len(self.source_node_set), len(self.destination_node_set))

        for lid in self.link_data.keys():

            if self.link_data[lid]['F_NODE'] in self.neighbors_list:
                self.neighbors_list[self.link_data[lid]['F_NODE']].append(self.link_data[lid]['T_NODE'])
            else:
                self.neighbors_list[self.link_data[lid]['F_NODE']] = [self.link_data[lid]['T_NODE']]

            if (self.link_data[lid]['F_NODE'], self.link_data[lid]['T_NODE']) in self.link_pair_data:
                self.link_pair_data[(self.link_data[lid]['F_NODE']), self.link_data[lid]['T_NODE']].append(lid)
            else:
                self.link_pair_data[(self.link_data[lid]['F_NODE']), self.link_data[lid]['T_NODE']] = [lid]

    def reset_traffic_info(self):

        for l in self.link_data.keys():
            # print('before', self.traffic_info[l])
            maxspd = self.link_data[l]['MAX_SPD']
            self.traffic_info[l] = list(np.random.random_integers(maxspd-maxspd*0.3, maxspd, 288))
            # self.traffic_info[l] = list(np.ones(288) * maxspd)
            # print('after', self.traffic_info[l])




    def get_link_id(self, fnode, tnode):
        return len(self.link_pair_data[(fnode,tnode)]), self.link_pair_data[(fnode,tnode)]

    def get_node_id(self, idx):
        # print(self.node_data[idx])
        return self.node_data[idx]['NODE_ID']

    def neighbors(self, fnode):                 #  return
        # pp.pprint(self.neighbors_list)
        return self.neighbors_list[fnode]

    def weight(self, link_id):
        # link_id =self.get_link_id(fromnode, tonode)
        return self.link_data[link_id]['WEIGHT']

    def weight(self, fromnode, tonode):
        n, link_id_list = self.get_link_id(fromnode, tonode)
        lid = link_id_list[0]
        return self.link_data[link_id_list[0]]['WEIGHT']

    def distance(self, link_id):
        # link_id = self.get_link_id(fromnode, tonode)
        return self.link_data[link_id]['LENGTH']

    def distance(self, fromnode, tonode):
        # link_id = self.get_link_id(fromnode, tonode)
        n, link_id_list = self.get_link_id(fromnode, tonode)

        return self.link_data[link_id_list[0]]['LENGTH']

    def velocity(self, link_id):
        # link_id = self.get_link_id(fromnode, tonode)
        return self.link_data[link_id]['CUR_SPD']

    def velocity(self, fromnode, tonode, tidx):
        # link_id = self.get_link_id(fromnode, tonode)
        n, link_id_list = self.get_link_id(fromnode, tonode)
        link_id = link_id_list[0]
        return self.traffic_info[link_id][tidx]

    def nodes_xy(self, nidx):
        return (self.node_data[nidx]['long'], self.node_data[nidx]['lat'])

    def get_path_distance(self, path):
        distance=0
        for i in range(len(path)-1):
            fromenode = path[i]
            tonode = path[i+1]
            distance = distance+self.distance(fromenode, tonode)
        return distance

    def get_path_weight(self, path):
        weight = 0
        for i in range(len(path) - 1):
            fromenode = path[i]
            tonode = path[i + 1]
            weight = weight + self.weight(fromenode, tonode)
        return weight

    def get_path_avg_velo(self, path, tidx):
        sumvelo = 0
        for i in range(len(path) - 1):
            fromenode = path[i]
            tonode = path[i + 1]
            sumvelo = sumvelo + self.velocity(fromenode, tonode, tidx)
        return sumvelo/(len(path) - 1)

    def get_path_drivingtime(self, path, tidx):
        dtime = 0.0

        for i in range(len(path) - 1):
            fromenode = path[i]
            tonode = path[i + 1]
            dtime = dtime + self.distance(fromenode, tonode) / self.velocity(fromenode, tonode, tidx)

        return dtime

class Graph_simple_6:
    def __init__(self):
        self.link_data, self.node_data, self.traffic_info, self.cs_info, self.minx, self.miny, self.maxx, self.maxy = data_gen.network_info_simple_6()
        # self.link_data, self.node_data, self.traffic_info, self.cs_info, self.minx, self.miny, self.maxx, self.maxy = data_gen.network_info_simple_25()
        self.num_node =  len(self.node_data)
        self.num_link = len(self.link_data)
        self.neighbors_list = {}
        self.link_pair_data = {}
        self.source_node_set = set()
        self.destination_node_set = set()

        # print('link 4070105400', self.link_data[4070105400])

        print('Num links from link data', len(self.link_data.keys()))
        print('Num node from link data', len(self.node_data.keys()))
        count = 0
        errorlink = 0

        for l in self.link_data.keys():
            maxspd = self.link_data[l]['MAX_SPD']
            self.traffic_info[l] = list(np.random.random_integers(maxspd-maxspd*0.3, maxspd, 288))
            # self.traffic_info[l] = list(np.ones(288)*maxspd)




        print('Num links from traffic data', len(self.traffic_info.keys()))
        print('Modified Num link', len(self.link_data.keys()))

        for l in self.link_data.keys():
            self.source_node_set.add(self.link_data[l]['F_NODE'])
            self.destination_node_set.add(self.link_data[l]['T_NODE'])

        # print(len(self.node_data), len(self.source_node_set), len(self.destination_node_set))

        for lid in self.link_data.keys():

            if self.link_data[lid]['F_NODE'] in self.neighbors_list:
                self.neighbors_list[self.link_data[lid]['F_NODE']].append(self.link_data[lid]['T_NODE'])
            else:
                self.neighbors_list[self.link_data[lid]['F_NODE']] = [self.link_data[lid]['T_NODE']]

            if (self.link_data[lid]['F_NODE'], self.link_data[lid]['T_NODE']) in self.link_pair_data:
                self.link_pair_data[(self.link_data[lid]['F_NODE']), self.link_data[lid]['T_NODE']].append(lid)
            else:
                self.link_pair_data[(self.link_data[lid]['F_NODE']), self.link_data[lid]['T_NODE']] = [lid]

    def reset_traffic_info(self):

        for l in self.link_data.keys():
            # print('before', self.traffic_info[l])
            maxspd = self.link_data[l]['MAX_SPD']
            self.traffic_info[l] = list(np.random.random_integers(maxspd-maxspd*0.3, maxspd, 288))
            # print('after', self.traffic_info[l])




    def get_link_id(self, fnode, tnode):
        return len(self.link_pair_data[(fnode,tnode)]), self.link_pair_data[(fnode,tnode)]

    def get_node_id(self, idx):
        # print(self.node_data[idx])
        return self.node_data[idx]['NODE_ID']

    def neighbors(self, fnode):                 #  return
        # pp.pprint(self.neighbors_list)
        return self.neighbors_list[fnode]

    def weight(self, link_id):
        # link_id =self.get_link_id(fromnode, tonode)
        return self.link_data[link_id]['WEIGHT']

    def weight(self, fromnode, tonode):
        n, link_id_list = self.get_link_id(fromnode, tonode)
        lid = link_id_list[0]
        return self.link_data[link_id_list[0]]['WEIGHT']

    def distance(self, link_id):
        # link_id = self.get_link_id(fromnode, tonode)
        return self.link_data[link_id]['LENGTH']

    def distance(self, fromnode, tonode):
        # link_id = self.get_link_id(fromnode, tonode)
        n, link_id_list = self.get_link_id(fromnode, tonode)

        return self.link_data[link_id_list[0]]['LENGTH']

    def velocity(self, link_id):
        # link_id = self.get_link_id(fromnode, tonode)
        return self.link_data[link_id]['CUR_SPD']

    def velocity(self, fromnode, tonode, tidx):
        # link_id = self.get_link_id(fromnode, tonode)
        n, link_id_list = self.get_link_id(fromnode, tonode)
        link_id = link_id_list[0]
        return self.traffic_info[link_id][tidx]

    def nodes_xy(self, nidx):
        return (self.node_data[nidx]['long'], self.node_data[nidx]['lat'])

    def get_path_distance(self, path):
        distance=0
        for i in range(len(path)-1):
            fromenode = path[i]
            tonode = path[i+1]
            distance = distance+self.distance(fromenode, tonode)
        return distance

    def get_path_weight(self, path):
        weight = 0
        for i in range(len(path) - 1):
            fromenode = path[i]
            tonode = path[i + 1]
            weight = weight + self.weight(fromenode, tonode)
        return weight

    def get_path_avg_velo(self, path, tidx):
        sumvelo = 0
        for i in range(len(path) - 1):
            fromenode = path[i]
            tonode = path[i + 1]
            sumvelo = sumvelo + self.velocity(fromenode, tonode, tidx)
        return sumvelo/(len(path) - 1)

    def get_path_drivingtime(self, path, tidx):
        dtime = 0.0

        for i in range(len(path) - 1):
            fromenode = path[i]
            tonode = path[i + 1]
            dtime = dtime + self.distance(fromenode, tonode) / self.velocity(fromenode, tonode, tidx)

        return dtime

class Graph_jeju:
    def __init__(self, datapath):
        self.link_data, self.node_data, self.traffic_info, self.cs_info, self.minx, self.miny, self.maxx, self.maxy = data_gen.network_info(datapath)
        self.num_node =  len(self.node_data)
        self.num_link = len(self.link_data)
        self.neighbors_list = {}
        self.link_pair_data = {}
        self.source_node_set = set()
        self.destination_node_set = set()

        # print('link 4070105400', self.link_data[4070105400])

        print('Num links from link data', len(self.link_data.keys()))
        print('Num node from link data', len(self.node_data.keys()))
        count = 0
        errorlink = 0

        for l in self.link_data.keys():

            if l in self.traffic_info.keys():
                count += 1

                if len(self.traffic_info[l]) < 288:
                    avg = sum(self.traffic_info[l]) / len(self.traffic_info[l])
                    for i in range(288 - len(self.traffic_info[l])):
                        self.traffic_info[l].append(int(avg))
                    errorlink += 1
            else:
                maxspd = self.link_data[l]['MAX_SPD']
                self.traffic_info[l] = list(np.random.random_integers(maxspd-maxspd*0.3, maxspd, 288))




        print('Num links from traffic data', len(self.traffic_info.keys()))
        print('Modified Num link', len(self.link_data.keys()))

        for l in self.link_data.keys():
            self.source_node_set.add(self.link_data[l]['F_NODE'])
            self.destination_node_set.add(self.link_data[l]['T_NODE'])

        # print(len(self.node_data), len(self.source_node_set), len(self.destination_node_set))

        for lid in self.link_data.keys():

            if self.link_data[lid]['F_NODE'] in self.neighbors_list:
                self.neighbors_list[self.link_data[lid]['F_NODE']].append(self.link_data[lid]['T_NODE'])
            else:
                self.neighbors_list[self.link_data[lid]['F_NODE']] = [self.link_data[lid]['T_NODE']]

            if (self.link_data[lid]['F_NODE'], self.link_data[lid]['T_NODE']) in self.link_pair_data:
                self.link_pair_data[(self.link_data[lid]['F_NODE']), self.link_data[lid]['T_NODE']].append(lid)
            else:
                self.link_pair_data[(self.link_data[lid]['F_NODE']), self.link_data[lid]['T_NODE']] = [lid]

    def reset_traffic_info(self):
        for l in self.traffic_info.keys():
            # print('before', self.traffic_info[l])
            maxspd = self.link_data[l]['MAX_SPD']
            for tidx in range(288):
                curspd = self.traffic_info[l][tidx]
                newspd = int(np.random.normal(curspd, curspd*0.05))
                while newspd<=0:
                    newspd = int(np.random.normal(curspd, curspd * 0.05))
                self.traffic_info[l][tidx] = newspd
            # print('after', self.traffic_info[l])


    def get_link_id(self, fnode, tnode):
        return len(self.link_pair_data[(fnode,tnode)]), self.link_pair_data[(fnode,tnode)]

    def get_node_id(self, idx):
        # print(self.node_data[idx])
        return self.node_data[idx]['NODE_ID']

    def neighbors(self, fnode):                 #  return
        # pp.pprint(self.neighbors_list)
        return self.neighbors_list[fnode]

    def weight(self, link_id):
        # link_id =self.get_link_id(fromnode, tonode)
        return self.link_data[link_id]['WEIGHT']

    def weight(self, fromnode, tonode):
        n, link_id_list = self.get_link_id(fromnode, tonode)
        lid = link_id_list[0]
        return self.link_data[link_id_list[0]]['WEIGHT']

    def distance(self, link_id):
        # link_id = self.get_link_id(fromnode, tonode)
        return self.link_data[link_id]['LENGTH']

    def distance(self, fromnode, tonode):
        # link_id = self.get_link_id(fromnode, tonode)
        n, link_id_list = self.get_link_id(fromnode, tonode)

        return self.link_data[link_id_list[0]]['LENGTH']

    def velocity(self, link_id):
        # link_id = self.get_link_id(fromnode, tonode)
        return self.link_data[link_id]['CUR_SPD']

    def velocity(self, fromnode, tonode, tidx):
        # link_id = self.get_link_id(fromnode, tonode)
        n, link_id_list = self.get_link_id(fromnode, tonode)
        link_id = link_id_list[0]
        return self.traffic_info[link_id][tidx]

    def nodes_xy(self, nidx):
        return (self.node_data[nidx]['long'], self.node_data[nidx]['lat'])

    def get_path_distance(self, path):
        distance=0
        for i in range(len(path)-1):
            fromenode = path[i]
            tonode = path[i+1]
            distance = distance+self.distance(fromenode, tonode)
        return distance

    def get_path_weight(self, path):
        weight = 0
        for i in range(len(path) - 1):
            fromenode = path[i]
            tonode = path[i + 1]
            weight = weight + self.weight(fromenode, tonode)
        return weight

    def get_path_avg_velo(self, path, tidx):
        sumvelo = 0
        for i in range(len(path) - 1):
            fromenode = path[i]
            tonode = path[i + 1]
            sumvelo = sumvelo + self.velocity(fromenode, tonode, tidx)
        return sumvelo/(len(path) - 1)

    def get_path_drivingtime(self, path, tidx):
        dtime = 0.0

        for i in range(len(path) - 1):
            fromenode = path[i]
            tonode = path[i + 1]
            dtime = dtime + self.distance(fromenode, tonode) / self.velocity(fromenode, tonode, tidx)

        return dtime


class Graph_jejusi:
    def __init__(self):
        self.link_data, self.node_data, self.traffic_info, self.cs_info, self.minx, self.miny, self.maxx, self.maxy = data_gen.network_info_jejusi()
        self.num_node =  len(self.node_data)
        self.num_link = len(self.link_data)
        self.neighbors_list = {}
        self.link_pair_data = {}
        self.source_node_set = set()
        self.destination_node_set = set()

        # print('link 4070105400', self.link_data[4070105400])

        print('Num links from link data', len(self.link_data.keys()))
        print('Num node from link data', len(self.node_data.keys()))
        count = 0
        errorlink = 0

        for l in self.link_data.keys():

            if l in self.traffic_info.keys():
                count += 1

                if len(self.traffic_info[l]) < 288:
                    avg = sum(self.traffic_info[l]) / len(self.traffic_info[l])
                    for i in range(288 - len(self.traffic_info[l])):
                        self.traffic_info[l].append(int(avg))
                    errorlink += 1
            else:
                maxspd = self.link_data[l]['MAX_SPD']
                self.traffic_info[l] = list(np.random.random_integers(maxspd-maxspd*0.3, maxspd, 288))




        print('Num links from traffic data', len(self.traffic_info.keys()))
        print('Modified Num link', len(self.link_data.keys()))

        for l in self.link_data.keys():
            self.source_node_set.add(self.link_data[l]['F_NODE'])
            self.destination_node_set.add(self.link_data[l]['T_NODE'])

        # print(len(self.node_data), len(self.source_node_set), len(self.destination_node_set))

        for lid in self.link_data.keys():

            if self.link_data[lid]['F_NODE'] in self.neighbors_list:
                self.neighbors_list[self.link_data[lid]['F_NODE']].append(self.link_data[lid]['T_NODE'])
            else:
                self.neighbors_list[self.link_data[lid]['F_NODE']] = [self.link_data[lid]['T_NODE']]

            if (self.link_data[lid]['F_NODE'], self.link_data[lid]['T_NODE']) in self.link_pair_data:
                self.link_pair_data[(self.link_data[lid]['F_NODE']), self.link_data[lid]['T_NODE']].append(lid)
            else:
                self.link_pair_data[(self.link_data[lid]['F_NODE']), self.link_data[lid]['T_NODE']] = [lid]

    def reset_traffic_info(self):
        for l in self.traffic_info.keys():
            # print('before', self.traffic_info[l])
            maxspd = self.link_data[l]['MAX_SPD']
            for tidx in range(288):
                curspd = self.traffic_info[l][tidx]
                newspd = int(np.random.normal(curspd, curspd*0.05))
                while newspd<=0:
                    newspd = int(np.random.normal(curspd, curspd * 0.05))
                self.traffic_info[l][tidx] = newspd
            # print('after', self.traffic_info[l])


    def get_link_id(self, fnode, tnode):
        return len(self.link_pair_data[(fnode,tnode)]), self.link_pair_data[(fnode,tnode)]

    def get_node_id(self, idx):
        # print(self.node_data[idx])
        return self.node_data[idx]['NODE_ID']

    def neighbors(self, fnode):                 #  return
        # pp.pprint(self.neighbors_list)
        return self.neighbors_list[fnode]

    def weight(self, link_id):
        # link_id =self.get_link_id(fromnode, tonode)
        return self.link_data[link_id]['WEIGHT']

    def weight(self, fromnode, tonode):
        n, link_id_list = self.get_link_id(fromnode, tonode)
        lid = link_id_list[0]
        return self.link_data[link_id_list[0]]['WEIGHT']

    def distance(self, link_id):
        # link_id = self.get_link_id(fromnode, tonode)
        return self.link_data[link_id]['LENGTH']

    def distance(self, fromnode, tonode):
        # link_id = self.get_link_id(fromnode, tonode)
        n, link_id_list = self.get_link_id(fromnode, tonode)

        return self.link_data[link_id_list[0]]['LENGTH']

    def velocity(self, link_id):
        # link_id = self.get_link_id(fromnode, tonode)
        return self.link_data[link_id]['CUR_SPD']

    def velocity(self, fromnode, tonode, tidx):
        # link_id = self.get_link_id(fromnode, tonode)
        n, link_id_list = self.get_link_id(fromnode, tonode)
        link_id = link_id_list[0]
        return self.traffic_info[link_id][tidx]

    def nodes_xy(self, nidx):
        return (self.node_data[nidx]['long'], self.node_data[nidx]['lat'])

    def get_path_distance(self, path):
        distance=0
        for i in range(len(path)-1):
            fromenode = path[i]
            tonode = path[i+1]
            distance = distance+self.distance(fromenode, tonode)
        return distance

    def get_path_weight(self, path):
        weight = 0
        for i in range(len(path) - 1):
            fromenode = path[i]
            tonode = path[i + 1]
            weight = weight + self.weight(fromenode, tonode)
        return weight

    def get_path_avg_velo(self, path, tidx):
        sumvelo = 0
        for i in range(len(path) - 1):
            fromenode = path[i]
            tonode = path[i + 1]
            sumvelo = sumvelo + self.velocity(fromenode, tonode, tidx)
        return sumvelo/(len(path) - 1)

    def get_path_drivingtime(self, path, tidx):
        dtime = 0.0

        for i in range(len(path) - 1):
            fromenode = path[i]
            tonode = path[i + 1]
            dtime = dtime + self.distance(fromenode, tonode) / self.velocity(fromenode, tonode, tidx)

        return dtime



class Graph_jejusi_small:
    def __init__(self):
        self.link_data, self.node_data, self.traffic_info, self.cs_info, self.minx, self.miny, self.maxx, self.maxy = data_gen.network_info_jejusi_small()
        self.num_node =  len(self.node_data)
        self.num_link = len(self.link_data)
        self.neighbors_list = {}
        self.link_pair_data = {}
        self.source_node_set = set()
        self.destination_node_set = set()

        # print('link 4070105400', self.link_data[4070105400])

        print('Num links from link data', len(self.link_data.keys()))
        print('Num node from link data', len(self.node_data.keys()))
        count = 0
        errorlink = 0

        for l in self.link_data.keys():

            if l in self.traffic_info.keys():
                count += 1

                if len(self.traffic_info[l]) < 288:
                    avg = sum(self.traffic_info[l]) / len(self.traffic_info[l])
                    for i in range(288 - len(self.traffic_info[l])):
                        self.traffic_info[l].append(int(avg))
                    errorlink += 1
            else:
                maxspd = self.link_data[l]['MAX_SPD']
                self.traffic_info[l] = list(np.random.random_integers(maxspd-maxspd*0.3, maxspd, 288))




        print('Num links from traffic data', len(self.traffic_info.keys()))
        print('Modified Num link', len(self.link_data.keys()))

        for l in self.link_data.keys():
            self.source_node_set.add(self.link_data[l]['F_NODE'])
            self.destination_node_set.add(self.link_data[l]['T_NODE'])

        # print(len(self.node_data), len(self.source_node_set), len(self.destination_node_set))

        for lid in self.link_data.keys():

            if self.link_data[lid]['F_NODE'] in self.neighbors_list:
                self.neighbors_list[self.link_data[lid]['F_NODE']].append(self.link_data[lid]['T_NODE'])
            else:
                self.neighbors_list[self.link_data[lid]['F_NODE']] = [self.link_data[lid]['T_NODE']]

            if (self.link_data[lid]['F_NODE'], self.link_data[lid]['T_NODE']) in self.link_pair_data:
                self.link_pair_data[(self.link_data[lid]['F_NODE']), self.link_data[lid]['T_NODE']].append(lid)
            else:
                self.link_pair_data[(self.link_data[lid]['F_NODE']), self.link_data[lid]['T_NODE']] = [lid]

    def reset_traffic_info(self):
        for l in self.traffic_info.keys():
            # print('before', self.traffic_info[l])
            maxspd = self.link_data[l]['MAX_SPD']
            for tidx in range(288):
                curspd = self.traffic_info[l][tidx]
                newspd = int(np.random.normal(curspd, curspd*0.05))
                while newspd<=0:
                    newspd = int(np.random.normal(curspd, curspd * 0.05))
                self.traffic_info[l][tidx] = newspd
            # print('after', self.traffic_info[l])


    def get_link_id(self, fnode, tnode):
        return len(self.link_pair_data[(fnode,tnode)]), self.link_pair_data[(fnode,tnode)]

    def get_node_id(self, idx):
        # print(self.node_data[idx])
        return self.node_data[idx]['NODE_ID']

    def neighbors(self, fnode):                 #  return
        # pp.pprint(self.neighbors_list)
        return self.neighbors_list[fnode]

    def weight(self, link_id):
        # link_id =self.get_link_id(fromnode, tonode)
        return self.link_data[link_id]['WEIGHT']

    def weight(self, fromnode, tonode):
        n, link_id_list = self.get_link_id(fromnode, tonode)
        lid = link_id_list[0]
        return self.link_data[link_id_list[0]]['WEIGHT']

    def distance(self, link_id):
        # link_id = self.get_link_id(fromnode, tonode)
        return self.link_data[link_id]['LENGTH']

    def distance(self, fromnode, tonode):
        # link_id = self.get_link_id(fromnode, tonode)
        n, link_id_list = self.get_link_id(fromnode, tonode)

        return self.link_data[link_id_list[0]]['LENGTH']

    def velocity(self, link_id):
        # link_id = self.get_link_id(fromnode, tonode)
        return self.link_data[link_id]['CUR_SPD']

    def velocity(self, fromnode, tonode, tidx):
        # link_id = self.get_link_id(fromnode, tonode)
        n, link_id_list = self.get_link_id(fromnode, tonode)
        link_id = link_id_list[0]
        return self.traffic_info[link_id][tidx]

    def nodes_xy(self, nidx):
        return (self.node_data[nidx]['long'], self.node_data[nidx]['lat'])

    def get_path_distance(self, path):
        distance=0
        for i in range(len(path)-1):
            fromenode = path[i]
            tonode = path[i+1]
            distance = distance+self.distance(fromenode, tonode)
        return distance

    def get_path_weight(self, path):
        weight = 0
        for i in range(len(path) - 1):
            fromenode = path[i]
            tonode = path[i + 1]
            weight = weight + self.weight(fromenode, tonode)
        return weight

    def get_path_avg_velo(self, path, tidx):
        sumvelo = 0
        for i in range(len(path) - 1):
            fromenode = path[i]
            tonode = path[i + 1]
            sumvelo = sumvelo + self.velocity(fromenode, tonode, tidx)
        return sumvelo/(len(path) - 1)

    def get_path_drivingtime(self, path, tidx):
        dtime = 0.0

        for i in range(len(path) - 1):
            fromenode = path[i]
            tonode = path[i + 1]
            dtime = dtime + self.distance(fromenode, tonode) / self.velocity(fromenode, tonode, tidx)

        return dtime


class Graph_jejusi_77:
    def __init__(self):
        self.link_data, self.node_data, self.traffic_info, self.cs_info, self.minx, self.miny, self.maxx, self.maxy = data_gen.network_info_jejusi_77()
        # self.link_data, self.node_data, self.traffic_info, self.cs_info, 140.8, 39.5, 141.2, 39.7 = data_gen.network_info_jejusi_77()
        self.num_node =  len(self.node_data)
        self.num_link = len(self.link_data)
        self.neighbors_list = {}
        self.link_pair_data = {}
        self.source_node_set = set()
        self.destination_node_set = set()

        # print('link 4070105400', self.link_data[4070105400])

        print('Num links from link data', len(self.link_data.keys()))
        print('Num node from link data', len(self.node_data.keys()))
        count = 0
        errorlink = 0

        for l in self.link_data.keys():

            if l in self.traffic_info.keys():
                count += 1

                if len(self.traffic_info[l]) < 288:
                    avg = sum(self.traffic_info[l]) / len(self.traffic_info[l])
                    for i in range(288 - len(self.traffic_info[l])):
                        self.traffic_info[l].append(int(avg))
                    errorlink += 1
            else:
                maxspd = self.link_data[l]['MAX_SPD']
                self.traffic_info[l] = list(np.random.random_integers(maxspd-maxspd*0.3, maxspd, 288))




        print('Num links from traffic data', len(self.traffic_info.keys()))
        print('Modified Num link', len(self.link_data.keys()))

        for l in self.link_data.keys():
            self.source_node_set.add(self.link_data[l]['F_NODE'])
            self.destination_node_set.add(self.link_data[l]['T_NODE'])

        # print(len(self.node_data), len(self.source_node_set), len(self.destination_node_set))

        for lid in self.link_data.keys():

            if self.link_data[lid]['F_NODE'] in self.neighbors_list:
                self.neighbors_list[self.link_data[lid]['F_NODE']].append(self.link_data[lid]['T_NODE'])
            else:
                self.neighbors_list[self.link_data[lid]['F_NODE']] = [self.link_data[lid]['T_NODE']]

            if (self.link_data[lid]['F_NODE'], self.link_data[lid]['T_NODE']) in self.link_pair_data:
                self.link_pair_data[(self.link_data[lid]['F_NODE']), self.link_data[lid]['T_NODE']].append(lid)
            else:
                self.link_pair_data[(self.link_data[lid]['F_NODE']), self.link_data[lid]['T_NODE']] = [lid]

    # def reset_traffic_info(self):
    #     for l in self.traffic_info.keys():
    #         # print('before', self.traffic_info[l])
    #         maxspd = self.link_data[l]['MAX_SPD']
    #         for tidx in range(288):
    #             curspd = self.traffic_info[l][tidx]
    #             newspd = int(np.random.normal(curspd, curspd*0.05))
    #             while newspd<=0:
    #                 newspd = int(np.random.normal(curspd, curspd * 0.05))
    #             self.traffic_info[l][tidx] = newspd
    #         # print('after', self.traffic_info[l])

    def reset_traffic_info(self):

        for l in self.link_data.keys():
            # print('before', self.traffic_info[l])
            maxspd = self.link_data[l]['MAX_SPD']
            self.traffic_info[l] = list(np.random.random_integers(maxspd-maxspd*0.3, maxspd, 288))
            # print('after', self.traffic_info[l])



    def get_link_id(self, fnode, tnode):
        return len(self.link_pair_data[(fnode,tnode)]), self.link_pair_data[(fnode,tnode)]

    def get_node_id(self, idx):
        # print(self.node_data[idx])
        return self.node_data[idx]['NODE_ID']

    def neighbors(self, fnode):                 #  return
        # pp.pprint(self.neighbors_list)
        return self.neighbors_list[fnode]

    def weight(self, link_id):
        # link_id =self.get_link_id(fromnode, tonode)
        return self.link_data[link_id]['WEIGHT']

    def weight(self, fromnode, tonode):
        n, link_id_list = self.get_link_id(fromnode, tonode)
        lid = link_id_list[0]
        return self.link_data[link_id_list[0]]['WEIGHT']

    def distance(self, link_id):
        # link_id = self.get_link_id(fromnode, tonode)
        return self.link_data[link_id]['LENGTH']

    def distance(self, fromnode, tonode):
        # link_id = self.get_link_id(fromnode, tonode)
        n, link_id_list = self.get_link_id(fromnode, tonode)

        return self.link_data[link_id_list[0]]['LENGTH']

    def velocity(self, link_id):
        # link_id = self.get_link_id(fromnode, tonode)
        return self.link_data[link_id]['CUR_SPD']

    def velocity(self, fromnode, tonode, tidx):
        # link_id = self.get_link_id(fromnode, tonode)
        n, link_id_list = self.get_link_id(fromnode, tonode)
        link_id = link_id_list[0]
        return self.traffic_info[link_id][tidx]

    def nodes_xy(self, nidx):
        return (self.node_data[nidx]['long'], self.node_data[nidx]['lat'])

    def get_path_distance(self, path):
        distance=0
        for i in range(len(path)-1):
            fromenode = path[i]
            tonode = path[i+1]
            distance = distance+self.distance(fromenode, tonode)
        return distance

    def get_path_weight(self, path):
        weight = 0
        for i in range(len(path) - 1):
            fromenode = path[i]
            tonode = path[i + 1]
            weight = weight + self.weight(fromenode, tonode)
        return weight

    def get_path_avg_velo(self, path, tidx):
        sumvelo = 0
        for i in range(len(path) - 1):
            fromenode = path[i]
            tonode = path[i + 1]
            sumvelo = sumvelo + self.velocity(fromenode, tonode, tidx)
        return sumvelo/(len(path) - 1)

    def get_path_drivingtime(self, path, tidx):
        dtime = 0.0

        for i in range(len(path) - 1):
            fromenode = path[i]
            tonode = path[i + 1]
            dtime = dtime + self.distance(fromenode, tonode) / self.velocity(fromenode, tonode, tidx)

        return dtime


class Graph_34:
    def __init__(self):
        # np.random.seed(1)

        self.link_data, self.node_data, self.traffic_info, self.cs_info, self.minx, self.miny, self.maxx, self.maxy = data_gen.network_info_34()
        # self.link_data, self.node_data, self.traffic_info, self.cs_info, 140.8, 39.5, 141.2, 39.7 = data_gen.network_info_jejusi_77()
        self.num_node =  len(self.node_data)
        self.num_link = len(self.link_data)
        self.num_cs = len(self.cs_info)
        self.neighbors_list = {}
        self.link_pair_data = {}
        self.source_node_set = set()
        self.destination_node_set = set()

        # print('link 4070105400', self.link_data[4070105400])

        print('Num links from link data', len(self.link_data.keys()))
        print('Num node from link data', len(self.node_data.keys()))
        count = 0
        errorlink = 0


        for l in self.link_data.keys():
            maxspd = self.link_data[l]['MAX_SPD']
            self.traffic_info[l] = list(np.full(288, maxspd))
            # self.traffic_info[l] = list(np.random.random_integers(maxspd-maxspd*0.3, maxspd, 288))
            # self.traffic_info[l] = list(np.ones(288)*maxspd)




        print('Num links from traffic data', len(self.traffic_info.keys()))
        print('Modified Num link', len(self.link_data.keys()))

        for l in self.link_data.keys():
            self.source_node_set.add(self.link_data[l]['F_NODE'])
            self.destination_node_set.add(self.link_data[l]['T_NODE'])

        # print(len(self.node_data), len(self.source_node_set), len(self.destination_node_set))

        for lid in self.link_data.keys():

            if self.link_data[lid]['F_NODE'] in self.neighbors_list:
                self.neighbors_list[self.link_data[lid]['F_NODE']].append(self.link_data[lid]['T_NODE'])
            else:
                self.neighbors_list[self.link_data[lid]['F_NODE']] = [self.link_data[lid]['T_NODE']]

            if (self.link_data[lid]['F_NODE'], self.link_data[lid]['T_NODE']) in self.link_pair_data:
                self.link_pair_data[(self.link_data[lid]['F_NODE']), self.link_data[lid]['T_NODE']].append(lid)
            else:
                self.link_pair_data[(self.link_data[lid]['F_NODE']), self.link_data[lid]['T_NODE']] = [lid]





    def reset_traffic_info(self):
        # np.random.seed(seed)
        for l in self.link_data.keys():
            # print('before', self.traffic_info[l])
            maxspd = self.link_data[l]['MAX_SPD']
            self.traffic_info[l] = list(np.full(288, maxspd))
            # self.traffic_info[l] = list(np.random.random_integers(maxspd-maxspd*0.3, maxspd, 288))
            # print('after', self.traffic_info[l])
        # print(self.traffic_info)



    def get_link_id(self, fnode, tnode):
        return len(self.link_pair_data[(fnode,tnode)]), self.link_pair_data[(fnode,tnode)]

    def get_node_id(self, idx):
        # print(self.node_data[idx])
        return self.node_data[idx]['NODE_ID']

    def neighbors(self, fnode):                 #  return
        # pp.pprint(self.neighbors_list)
        return self.neighbors_list[fnode]

    def weight(self, link_id):
        # link_id =self.get_link_id(fromnode, tonode)
        return self.link_data[link_id]['WEIGHT']

    def weight(self, fromnode, tonode):
        n, link_id_list = self.get_link_id(fromnode, tonode)
        lid = link_id_list[0]
        return self.link_data[link_id_list[0]]['WEIGHT']

    def distance(self, link_id):
        # link_id = self.get_link_id(fromnode, tonode)
        return self.link_data[link_id]['LENGTH']

    def distance(self, fromnode, tonode):
        # link_id = self.get_link_id(fromnode, tonode)
        n, link_id_list = self.get_link_id(fromnode, tonode)

        return self.link_data[link_id_list[0]]['LENGTH']

    # def velocity(self, link_id):
    #     # link_id = self.get_link_id(fromnode, tonode)
    #     return self.link_data[link_id]['CUR_SPD']

    def velocity(self, fromnode, tonode, tidx):
        # link_id = self.get_link_id(fromnode, tonode)
        n, link_id_list = self.get_link_id(fromnode, tonode)
        link_id = link_id_list[0]
        return self.traffic_info[link_id][tidx]

    def nodes_xy(self, nidx):
        return (self.node_data[nidx]['long'], self.node_data[nidx]['lat'])

    def get_path_distance(self, path):
        distance=0
        for i in range(len(path)-1):
            fromenode = path[i]
            tonode = path[i+1]
            distance = distance+self.distance(fromenode, tonode)
        return distance

    def get_path_weight(self, path):
        weight = 0
        for i in range(len(path) - 1):
            fromenode = path[i]
            tonode = path[i + 1]
            weight = weight + self.weight(fromenode, tonode)
        return weight

    def get_path_avg_velo(self, path, tidx):
        sumvelo = 0
        for i in range(len(path) - 1):
            fromenode = path[i]
            tonode = path[i + 1]
            sumvelo = sumvelo + self.velocity(fromenode, tonode, tidx)
        return sumvelo/(len(path) - 1)

    def get_path_drivingtime(self, path, tidx):
        dtime = 0.0

        for i in range(len(path) - 1):
            fromenode = path[i]
            tonode = path[i + 1]
            dtime = dtime + self.distance(fromenode, tonode) / self.velocity(fromenode, tonode, tidx)

        return dtime


if __name__ == "__main__":
    graph = Graph_simple_39()
    graph.reset_traffic_info()
