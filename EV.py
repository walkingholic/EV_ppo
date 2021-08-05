import setting

class EV:
    def __init__(self, id,  source, destination, soc, reqsoc, t_start):
        self.id = id
        self.t_start = t_start
        self.curr_time = t_start
        self.curr_day = 0

        self.curr_SOC = soc
        self.init_SOC = soc
        self.final_soc = reqsoc
        self.req_SOC = reqsoc
        self.before_charging_SOC=0.0
        self.source = source
        self.curr_location = source
        self.next_location = source
        self.destination = destination
        self.charged = 0
        self.cs = None
        self.csid = -1

        self.charging_effi = setting.charging_effi
        self.maxBCAPA= setting.maxBCAPA
        self.ECRate = setting.ECRate # kwh/km

        self.totalenergyconsumption = 0.0
        self.totaldrivingdistance = 0.0
        self.totaldrivingtime = 0.0

        self.expense_cost_part = 0.0
        self.expense_time_part = 0.0
        self.totalcost = 0.0

        self.true_arrtime = 0.0
        self.true_waitingtime = 0.0
        self.true_charging_duration = 0.0

        self.ept_arrtime = 0.0
        self.ept_waitingtime = 0.0
        self.ept_charging_duration = 0.0
        self.ept_totalcost = 0.0
        self.ept_totaldrivingtime = 0.0


        self.charging_finish_time = 0.0

        self.time_diff_WT = 0.0

        self.cschargingtime = 0.0
        self.cschargingcost = 0.0
        self.cschargingwaitingtime = 0.0
        self.cscharingenergy = 0.0
        self.cschargingprice = 0.0
        self.cschargingstarttime = 0.0
        self.eptcschargingstarttime = 0.0
        self.csdistance = 0
        self.csdrivingtime = 0
        self.cssoc = 0

        self.homechargingtime = 0.0
        self.homechargingcost = 0.0
        self.homechargingenergy = 0.0
        self.homechargingprice = 0.0
        self.homechargingstarttime = 0.0
        self.homedrivingdistance = 0.0
        self.homedrivingtime = 0.0
        self.homesoc = 0.0

        self.fdist=0
        self.rdist=0
        self.path=[]
        self.front_path = []
        self.rear_path = []

        self.predic_totaltraveltime = 0.0

        self.weightvalue = 0.0

    def print(self):
        print(self.id,  self.source, self.destination, self.t_start, self.curr_SOC)