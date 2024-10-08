import time
import os, sys
import numpy as np

from xml.etree.ElementTree import parse
from collections import defaultdict
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'],'tools')
    print(tools)
    sys.path.append(tools)
    import traci
else:
    sys.exit('Declare environment variable "SUMO_HOME"')
class dqnEnv():
    def __init__(self, sumoBinary, net_file: str, cfg_file: str, edgelists: list,alldets: list, dict_connection, 
                 veh:str, destination:str, state_size: int, action_size: int, config:dict, use_gui: bool = True,
            begin_time: int =0, num_seconds:int = 3600, max_depart_delay:int = 10000):
        
        self.sumoBinary = sumoBinary
        self.net = net_file
        # self.badpoints=badpoints

        
        self.sumocfg = cfg_file
        
        self.edgelists = edgelists
        self.alldets = alldets

        self.config = config
        self.use_gui = config['use_gui']

        self.veh = veh
        
        self.destination = destination

        self.episode = 0 # # of run time 
        self.begin_time = begin_time
        self.num_seconds = num_seconds
        self.max_depart_delay = max_depart_delay

        self.action_size = action_size
        self.state_size = state_size
        
        self.dict_connection = dict_connection
        self.sumo =traci

    def start_simulation(self):
        veh_id = int(self.config['veh_id'])
        if veh_id == 0:
            self.start_simulation_server()
        else:
            self.start_simulation_client()


    def start_simulation_client(self):
        self.sumo.init(port=int(self.config['port']),host = str(self.config['server_ip']))
        self.sumo.setOrder(int(self.config['veh_id'])+1)
        self.dict_edgelengths, self.list_edgelengths = self.get_edgelengths()
        self.dict_edgelimits = self.get_edgelimits()
        destlane = self.destination+'_0'
        self.destCord = self.sumo.lane.getShape(destlane)[0]


    def start_simulation_server(self):
        PORT = self.config['port']
        print("Sumo binary : {}".format(self.sumoBinary))
        print("Sumo config : {}".format(self.sumocfg))
        sumo_cmd = [self.sumoBinary,
            '-c', self.sumocfg,
            '--max-depart-delay', str(self.max_depart_delay)]
        if self.config['num_clients'] != 0:
            sumo_cmd.extend(['--num-clients',str(self.config['num_clients'])])
        if self.use_gui:
            if self.config['start']:
                sumo_cmd.extend(['--start'])
            if self.config['quit_on_end']:
                sumo_cmd.extend(['--quit-on-end'])
            self.sumo.start(sumo_cmd,PORT)
            self.sumo.setOrder(int(self.config['veh_id'])+1)
            self.sumo.gui.setSchema(self.sumo.gui.DEFAULT_VIEW, "real world")
            self.dict_edgelimits = self.get_edgelimits()
            destlane = self.destination+'_0'
            self.destCord = self.sumo.lane.getShape(destlane)[0]
        else:
            self.sumo.start(sumo_cmd,PORT)
            self.sumo.setOrder(int(self.config['veh_id'])+1)
            self.dict_edgelimits = self.get_edgelimits()
            destlane = self.destination+'_0'
            self.destCord = self.sumo.lane.getShape(destlane)[0]

        self.dict_edgelengths, self.list_edgelengths = self.get_edgelengths()

        '''
        if self.begin_time > 0:
            sumo_cmd.append('-b {}'.format(self.begin_time))
        
        '''
        
    # def start_simulation(self):
    #     sumo_cmd = [self.sumoBinary,
    #         '-c', self.sumocfg,
    #         '--max-depart-delay', str(self.max_depart_delay)]
    #     # if self.use_gui:
    #     #     sumo_cmd.extend(['--start', '--quit-on-end'])
    #         # traci.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")
        
    #     self.sumo.start(sumo_cmd)

    #     ## 모든 edge 별 length & speed limit 정보 얻어오기
    #     self.dict_edgelengths, self.list_edgelengths = self.get_edgelengths()
    
    #     self.dict_edgelimits = self.get_edgelimits()
    #     destlane = self.destination+'_0'
    #     self.destCord = self.sumo.lane.getShape(destlane)[0]
    #     #print('self.destCord: ',self.destCord)
    #     '''
    #     if self.begin_time > 0:
    #         sumo_cmd.append('-b {}'.format(self.begin_time))
    #         '''
        
    def sumo_step(self):
        self.sumo.simulationStep()

    def sumoclose(self):
        self.sumo.close()
        
    def reset(self):
        self.episode+=1 
        self.start_simulation()
        
        #vehicle  생성
        route_id = "rou"+self.veh[3:]
        id = int(self.veh[3:])
        route = self.config['start_edges']
        route = route[id]
        self.sumo.route.add(route_id, route) #default route
        self.sumo.vehicle.add(self.veh, route_id,departLane='best')
        curlane = self.get_curlane(self.veh)
        while curlane=='':
            curlane = self.get_curlane(self.veh)
            self.sumo.simulationStep()

        curedge = self.get_curedge(curlane)
        state = self.get_state(self.veh,curedge)
        return state

    def get_curlane(self,veh):
        curlane = self.sumo.vehicle.getLaneID(veh)
        return curlane

    def get_curedge(self,curlane): ##0217목 8pm 수정 오류 확인해봐야함  curlane ''뜨는거!!!!!!
        curedge = self.sumo.lane.getEdgeID(curlane)
        return curedge

    def get_done(self,curedge):
        done = False
        if curedge == self.destination:
            done = True
        return done
    
    def get_RoadID(self, veh):
        return traci.vehicle.getRoadID(veh)

    def get_reward(self, curedge, nextedge):
        reward = 0
        #reward = traveling time of curedge
        det = curedge.replace('E','D')
        num_veh = self.get_numVeh(det)
        if num_veh ==1: # 자기자신  #length/speedlimit if E7 or -E7 :10 else: 15
            traveltime = self.dict_edgelengths[curedge] / self.dict_edgelimits[curedge]
        else:
            #print('err1 get_reward num_veh 확인용 : ',num_veh) 
            traveltime = self.sumo.edge.getTraveltime(curedge) #(length/mean speed).
       
        reward = -traveltime
        return reward
        # reward=0
        # if nextedge in self.badpoints:
        #     reward = -100
        # elif nextedge=='E15':
        #     reward=500
        # return reward

    
    def get_nextedge(self, curedge, action):
        nextedge = self.dict_connection[curedge][action]
        return nextedge

    def get_numVeh(self, det):
        num_veh = self.sumo.lanearea.getLastStepVehicleNumber(det)
        return num_veh
    
    def get_edgelengths(self):
        dict_edgelengths = defaultdict(float)
        list_edgelengths = []
        dict_edgelengths.update((k,0.0 ) for k in self.edgelists)
        for edge in self.edgelists:
            lane = edge + '_0'
            length = self.sumo.lane.getLength(lane)
            dict_edgelengths[edge]=length
            list_edgelengths.append(length)
        return dict_edgelengths, list_edgelengths

    def get_edgelimits(self):
        dict_edgelimits = defaultdict(float)
        dict_edgelimits.update((k,15.0) for k in self.edgelists)
        dict_edgelimits['E7'] = 10.0
        dict_edgelimits['-E7'] = 10.0
        return dict_edgelimits

    def get_state(self, veh, curedge): #총 64개 원소
        state = []
        
        for edge in self.edgelists: #20개 edge별 차량수 
            det = edge.replace("E","D")
            state.append(self.get_numVeh(det)) 
        
        for edge in self.edgelists: #20개 edge별 평균 속도      
            state.append(self.sumo.edge.getLastStepMeanSpeed(edge)) 
        state.extend(self.list_edgelengths) #20개 edge length
       
        curlane = curedge+'_0' #curedge좌표
        curCord= self.sumo.lane.getShape(curlane)[0]
        state.extend(list(curCord))
        
        state.extend(list(self.destCord))#목적지 좌표
        
        return state

    def get_nextstate(self, veh, nextedge):
        next_state = []
        for edge in self.edgelists: #20개 edge별 차량수 
            det = edge.replace("E","D")
            next_state.append(self.get_numVeh(det)) 
            
        for edge in self.edgelists: #20개 edge별 평균 속도
            next_state.append(self.sumo.edge.getLastStepMeanSpeed) 
        next_state.append(self.list_edgelengths) #20개 edge length
        nextlane = nextedge+'_0' #nextedge좌표
        nextCord= self.sumo.lane.getShape(nextlane)[0]
        next_state.extend(list(nextCord))
        next_state.extend(list(self.destCord))
        return next_state

    def step(self, curedge, nextedge):
        
        beforeedge = curedge #비교해서 변하면 고를려고!

        done = self.get_done(curedge)
        reward = self.get_reward(curedge, nextedge)

        if done:
            return reward, done
        
        self.sumo.vehicle.changeTarget(self.veh,nextedge) #차량 움직여!
        
        while self.sumo.simulation.getMinExpectedNumber() > 0:
            #curlane = self.get_curlane(self.veh)
            
            #curedge = self.get_curedge(curlane)
            curedge = self.get_RoadID(self.veh)
            done = self.get_done(curedge)
            if done:
                break  
            self.sumo.simulationStep() 
            if curedge in self.edgelists and curedge !=beforeedge : #변했네!! 그럼 이제 다음 꺼 고르러 가야지
                break

        return reward, done    

    
    
    