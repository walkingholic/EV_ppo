import setting
import numpy as np
import matplotlib.pyplot as plt
import copy
import MainLogic as ta
import random
from EV import EV
import setting as st

# seed = st.random_seed
# np.random.seed(seed)
# random.seed(seed)

class CS:
    def __init__(self, node_id, profit, long, lat):
        # np.random.seed(seed)

        self.id = node_id
        self.price = list()
        self.waittime = list()
        self.fastchargingpower = setting.cs_chargingpower_fast # kw
        self.slowchargingpower = setting.cs_chargingpower_slow
        self.num_sorket = setting.N_SOCKET
        self.cplist = []


        self.unknown_ev = [] # pair (pev, true arr)
        self.charing_ev = [] # pair (pev, start, endtime)
        self.waiting_ev = []
        self.reserve_ev = [] # pair (pev, arrtime, endtime)
        self.x = long
        self.y = lat
        self.profit = profit
        self.TOU_price = [0.15575 for i in range(24)]

        for i in range(288):
            p = np.random.normal(self.TOU_price[int(i/12)], 0.30 * self.TOU_price[int(i/12)])
            while p < 0:
                p = np.random.normal(self.TOU_price[int(i/12)], 0.30 * self.TOU_price[int(i/12)])
            self.price.append(p*self.profit)

        for i in range(self.num_sorket):
            cp = self.CPOLE(i, self.id, self.price)
            self.cplist.append(cp)

    def update_avail_time(self, cur_time, graph):#현재시간을 기준으로 도착한 예약들을 처리.

        self.reserve_ev.sort(key=lambda element: element[2])
        for rev, eptTarr, _ in self.reserve_ev:
            # print('befor reserve_ev list CS ID:', self.id, 'reseved ev:', rev.id, 'eptTarr:', eptTarr, 'TruTarr:', rev.true_arrtime,'curTime:', cur_time)
            if rev.true_arrtime <= cur_time:
                self.waiting_ev.append((rev, rev.true_arrtime))
                self.reserve_ev = self.reserve_ev[1:]

        for uev, uevtarr in self.unknown_ev:
            if uevtarr < cur_time:
                self.waiting_ev.append((uev, uev.true_arrtime))
                self.unknown_ev = self.unknown_ev[1:]

        self.waiting_ev.sort(key=lambda e: e[1])

        tmpcp = []

        for cp in self.cplist:
            tmpcp.append((cp, cp.avail_time))

        for i, tp_wev in enumerate(self.waiting_ev):
            tmpcp.sort(key=lambda element: element[1])
            cp, at = tmpcp[0]
            wev, true_arrtime = tp_wev
            cp.set_charging(wev)
            ta.finishi_trip(wev, cp, graph)
            tmpcp[0] = cp, cp.avail_time

        self.waiting_ev = []

    def get_ept_avail_time(self, ept_arrtime): # 예상 정보를 가지고
        self.reserve_ev.sort(key=lambda element: element[1])
        tmp_rsv = copy.deepcopy(self.reserve_ev)
        tmp_wtev_list = []
        ept_extedWT_list = []
        for rev, eptTarr,_ in tmp_rsv:
            # print('CS ID:',self.id, 'reseved ev:', rev.id, 'eptTarr:', eptTarr,'TruTarr:', rev.true_arrtime)
            if eptTarr <= ept_arrtime:
                tmp_wtev_list.append((rev, eptTarr))
            else:
                # print('need to recalculate for charged WT')
                ept_extedWT_list.append((rev, eptTarr))

        tmp_wtev_list.sort(key=lambda e:e[1])
        tmp_atime = []
        for cp in self.cplist:
            tmp_atime.append(cp.avail_time)

        for i, tp_wev in enumerate(tmp_wtev_list):
            tmp_atime.sort()
            at = tmp_atime[0]
            wev, eptTarr = tp_wev
            if at < wev.ept_arrtime:
                eptWT = 0.0
                # wev.eptcschargingstarttime = wev.ept_arrtime + eptWT
                at = wev.ept_arrtime + wev.ept_charging_duration
            else:
                eptWT = at - wev.ept_arrtime
                # wev.eptcschargingstarttime = wev.ept_arrtime + eptWT
                at = at + wev.ept_charging_duration
            tmp_atime[0] = at

        return tmp_atime, ept_extedWT_list

    def get_ept_WT(self, pev, ept_arrtime, ept_charging_duration, cur_time, graph):

        self.reserve_ev.sort(key=lambda element: element[1])
        self.update_avail_time(cur_time, graph)
        at_list, ept_extedWT_list = self.get_ept_avail_time(ept_arrtime)  # charging finish time for ev being charged

        at_list.sort()
        minAT = at_list[0]
        if minAT > ept_arrtime:
            eptWT = minAT - ept_arrtime
        else:
            eptWT = 0

        # at_list[0] = at_list[0]+ept_charging_duration

        tmp_atime = copy.deepcopy(at_list)
        tmp_atime[0] = tmp_atime[0] + ept_charging_duration

        diff_ch_start = []
        ept_extedWT_list.sort(key=lambda e: e[1])
        for i, (wev, _) in enumerate(ept_extedWT_list):
            tmp_atime.sort()
            at = tmp_atime[0]

            if at < wev.ept_arrtime:
                eptWT = 0.0
                ept_ch_start = wev.ept_arrtime + eptWT
                diff = ept_ch_start - wev.eptcschargingstarttime
                diff_ch_start.append(diff)
                # wev.eptcschargingstarttime = ept_ch_start
                at = wev.ept_arrtime + wev.ept_charging_duration
            else:
                eptWT = at - wev.ept_arrtime
                ept_ch_start = wev.ept_arrtime + eptWT
                diff = ept_ch_start - wev.eptcschargingstarttime
                diff_ch_start.append(diff)
                # wev.eptcschargingstarttime = ept_ch_start
                at = at + wev.ept_charging_duration
            tmp_atime[0] = at

        return eptWT, at_list, sum(diff_ch_start)



    def get_current_WT(self, pev, cur_time, graph):

        self.reserve_ev.sort(key=lambda element: element[1])
        self.update_avail_time(cur_time, graph)

        wt_list = []
        at_list = []
        for cp in self.cplist:
            at_list.append(cp.avail_time)
            if cp.avail_time < cur_time:
                wt_list.append(0)
            else:
                wt_list.append(cp.avail_time-cur_time)

        wt_list.sort()
        currWT = wt_list[0]

        return currWT, at_list, 0



    def recieve_request(self, ev):
        self.reserve_ev.append((ev, ev.ept_arrtime, ev.true_arrtime))

        self.reserve_ev.sort(key=lambda e:e[1])

        tmp_wtev_list = []
        ept_extedWT_list = []

        for rev, eptTarr, _ in self.reserve_ev:
            # print('CS ID:',self.id, 'reseved ev:', rev.id, 'eptTarr:', eptTarr,'TruTarr:', rev.true_arrtime)
            if eptTarr <= ev.ept_arrtime:
                tmp_wtev_list.append((rev, eptTarr))
            else:
                # print('need to recalculate for charged WT')
                ept_extedWT_list.append((rev, eptTarr))

        diff_ch_start = []
        tmp_atime = []
        for cp in self.cplist:
            tmp_atime.append(cp.avail_time)

        for i, (wev, _, _) in enumerate(self.reserve_ev):
            tmp_atime.sort()
            at = tmp_atime[0]

            if at < wev.ept_arrtime:
                eptWT = 0.0
                ept_ch_start = wev.ept_arrtime + eptWT
                diff = ept_ch_start - wev.eptcschargingstarttime
                # diff_ch_start.append(diff)
                # if diff < 0:
                #     print('at < wev.ept_arrtime  ', at , wev.ept_arrtime, wev.ept_waitingtime)
                #
                #     print('이것은 상관없을꺼야, 0이겟지?  ', diff, ept_ch_start, wev.eptcschargingstarttime)
                # wev.eptcschargingstarttime = ept_ch_start
                at = wev.ept_arrtime + wev.ept_charging_duration
            else:
                eptWT = at - wev.ept_arrtime
                ept_ch_start = wev.ept_arrtime + eptWT
                diff = ept_ch_start - wev.eptcschargingstarttime

                if diff > 0:
                    diff_ch_start.append(diff)
                # print('at < wev.ept_arrtime  ', at, eptWT, wev.ept_arrtime, wev.ept_waitingtime)
                # print('이것은 상관있어', diff, ept_ch_start, wev.eptcschargingstarttime)
                # wev.eptcschargingstarttime = ept_ch_start
                at = at + wev.ept_charging_duration
            tmp_atime[0] = at

        # for i, (rev, _, _) in enumerate(self.reserve_ev):
        #     print(ev.curr_time, rev.id, diff_ch_start[i])

        return sum(diff_ch_start)



    def sim_finish(self, graph):
        self.update_avail_time(3000, graph)

    def get_price(self, sim_time):
        return self.price[int(sim_time/5)]


    class CPOLE:
        def __init__(self, id, csid, price):
            self.id = id
            self.csid = csid
            self.avail_time = 0.0
            self.charging_ev = []
            self.curr_charging_ev = None
            self.chargingpower = 60 #kw
            self.price = price

        def update_cpole(self, cur_time):
            if self.avail_time < cur_time:
                self.avail_time = cur_time
                self.curr_charging_ev = None

        def set_charging(self, ev):
            # print('set charging EV:{} CS:{}, CP:{}'.format(ev.id, self.csid, self.id))

            charging_energy = ev.maxBCAPA * (ev.req_SOC - ev.curr_SOC)
            charging_duration = (charging_energy/(self.chargingpower * ev.charging_effi))
            ev.cscharingenergy = charging_energy
            ev.true_charging_duration = charging_duration*60
            ev.cschargingtime = charging_duration*60

            self.charging_ev.append(ev)
            self.curr_charging_ev = ev
            # print("  1===setCharging AvailTime:", self.avail_time)
            if self.avail_time < ev.true_arrtime:
                ev.true_waitingtime = 0.0
                ev.cschargingstarttime = ev.true_arrtime + ev.true_waitingtime
                self.avail_time = ev.true_arrtime + ev.true_charging_duration
            else:
                ev.true_waitingtime = self.avail_time - ev.true_arrtime
                # print(ev.id, 'true_waitingtime: ', ev.true_waitingtime, 'avail_time: ',self.avail_time, 'true_arrtime: ',ev.true_arrtime)
                ev.cschargingstarttime = self.avail_time
                self.avail_time = self.avail_time + ev.true_charging_duration

            ev.time_diff_WT = ev.true_waitingtime - ev.ept_waitingtime
            ev.curr_time = ev.cschargingstarttime + ev.true_charging_duration
            ev.charging_finish_time = ev.curr_time
            ev.curr_SOC = ev.req_SOC
            ev.cschargingprice = self.price[int(ev.cschargingstarttime%1440/5)]
            ev.charged = 1
            # print("  2===setCharging AvailTime:", self.avail_time)
            # print("  3===setCharging EVID: {0:.2f}  truWT: {1:.2f}  truChDur: {2:.2f}".format(ev.id, ev.true_waitingtime, ev.true_charging_duration))

def reset_CS_info(graph):


    CS_list = []
    num_cs = len(graph.cs_info)
    # num_cs = N_CS
    profit = np.full(num_cs,0.7)


    for i, l in enumerate(graph.cs_info):
        cs = CS(l, profit[i], graph.cs_info[l]['long'], graph.cs_info[l]['lat'])

        CS_list.append(cs)
    return CS_list

