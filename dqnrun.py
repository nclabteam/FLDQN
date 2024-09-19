
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import time
import optparse
import random
import numpy as np
import matplotlib.pyplot as plt
from xml.etree.ElementTree import parse
import argparse
from collections import defaultdict
from dqnTrainedAgent import dqnTrainedAgent
from utils import *
from dqnenv import dqnEnv
from dqnagent import dqnAgent
import flwr as fl 
import tensorflow as tf 

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'],'tools')
    print(tools)
    sys.path.append(tools)
else:
    sys.exit('Declare environment variable "SUMO_HOME"')

from sumolib import checkBinary

def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("-N","--num_episode", 
                        default=100, help="numer of episode to run qlenv")
    optParser.add_option("--nogui", action="store_true",
                        default=False, help="run commandline version of sumo")
    optParser.add_option("--noplot", action="store_true",
                        default=False, help="save result in png")    
    optParser.add_option("--trained", "-T", action="store_true",
                        default=False, help="save result in png")    
    optParser.add_option("--config",type=str,default = "config.yaml")
    optParser.add_argument("--vehicle",type=str)                                
    options, args = optParser.parse_args()
    return options

def get_toedges(net, fromedge):
    #calculate reachable nextedges
    tree = parse(net)
    root = tree.getroot()
    toedges = []
    for connection in root.iter("connection"):
        if connection.get("from")==fromedge:
            toedges.append(connection.get("to"))
    return toedges

def get_alledges(net):
    #get plain edges by parsing net.xml
    tree = parse(net)
    root = tree.getroot()
    alledgelists = root.findall("edge")
    edgelists = [edge.get("id") for edge in alledgelists if ':' not in edge.get("id")]
    return edgelists

def get_edgesinfo(net):
    tree = parse(net)
    root = tree.getroot()
    alledgelists = root.findall("edge")
    edgesinfo = [x.find("lane").attrib for x in alledgelists]
    return edgesinfo

def calculate_connections(edgelists, net):
    # calculate dictionary of reachable edges(next edge) for every edge  
    tree = parse(net)
    root = tree.getroot()
    
    dict_connection = defaultdict(list)
    dict_connection.update((k,[]) for k in edgelists)

    for connection in root.iter("connection"):
        curedge = connection.get("from")
        if ':' not in curedge:
            dict_connection[curedge].append(connection.get("to"))

    for k,v in dict_connection.items():
        if len(v)==0:
            dict_connection[k]=['','','']
        elif len(v)==1:
            dict_connection[k].append('')
            dict_connection[k].append('')
        elif len(v)==2:
            dict_connection[k].append('')
    return dict_connection 


def generate_lanedetectionfile(net, det):
    #generate det.xml file by setting a detector at the end of each lane (-10m)
    alledges = get_alledges(net)
    edgesinfo = get_edgesinfo(net)
    alllanes = [edge +'_0' for edge in alledges]
    alldets =  [edge.replace("E","D") for edge in alledges]  
  
    with open(det,"w") as f:
        print('<additional>', file = f)
        for i,v in enumerate(edgesinfo):
            
            print('        <laneAreaDetector id="%s" lane="%s" pos="0.0" length="%s" freq ="%s" file="dqn_detfile.out"/>'
            %(alldets[i], v['id'],v['length'],"1"), file = f)
        print('</additional>', file = f)
    return alldets

def get_alldets(alledges):
    alldets =  [edge.replace("E","D") for edge in alledges]
    return alldets

def plot_result(num_seed, episodes, scores, dirResult, num_episode):
    pylab.plot(episodes, scores, 'b')
    pylab.xlabel('episode')
    pylab.ylabel('Mean Travel Time')
    pylab.savefig(dirResult+str(num_episode)+'_'+str(num_seed)+'.png')    

def plot_trainedresult(num_seed, episodes, scores, dirResult, num_episode):
    pylab.plot(episodes, scores, 'b')
    pylab.xlabel('episode')
    pylab.ylabel('Mean Travel Time')
    pylab.savefig(dirResult+'TrainedModel'+str(num_episode)+'_'+str(num_seed)+'.png')  

import time 
start_time=time.time()

#DQN routing : routing by applying DQN algorithm (using qlEnv & alAgent)
def dqn_run(agent,num_seed, trained,sumoBinary,plotResult, num_episode,net, trip, randomrou, add,dirResult,dirModel,
             sumocfg,fcdoutput, edgelists,alldets, dict_connection,veh,destination,  state_size, action_size, dqn_config,file_path,rounds=0):  
    env = dqnEnv(sumoBinary, net_file = net, cfg_file = sumocfg, edgelists = edgelists, alldets=alldets, 
                 dict_connection=dict_connection, veh = veh, destination = destination, state_size = state_size, action_size= action_size,config=dqn_config)
    if rounds > 0:
        agent = agent
    else:
        if trained :
            agent = dqnTrainedAgent(veh, num_seed, edgelists, dict_connection, state_size, action_size,num_episode, dirModel)
            print('**** [TrainedAgent {} Route Start] ****'.format(num_episode))
        else:
            agent = dqnAgent(veh,num_seed, edgelists, dict_connection, state_size, action_size, num_episode, dirModel)
    
    start = time.time()
    cntSuccess=0
    lst_cntSuccess=[]
    idxSuccess=-1
    scores, episodes = [],[]
    episode_time=[]
    simu_time = []
    score_avg = 0
    i_new=0
    for episode in range(num_episode):
        st=time.time()
        print("\n********#{} episode start***********".format(episode))
        #reset environment
        #generate random output 
        if veh == "veh0":
            path = os.path.join(os.environ['SUMO_HOME'],'tools','randomTrips.py')
            # path='"C:\\Program Files (x86)\\Eclipse\\Sumo\\tools\\randomTrips.py"'
            cmd_genDemand = "python {} -n {} -o {} -r {} -b 0 -e 3600 --period 3 --additional-file {} --trip-attributes  \"departLane='best' type='type1' departSpeed='max' departPos='random'\"  --random".format(path, net, trip, randomrou, add)    
            # print(cmd_genDemand)
            os.system(cmd_genDemand) 

        score = 0
        routes = []
        state = env.reset() 
        
        state = np.reshape(state,[1,state_size]) 
        curedge = env.get_RoadID(veh)
        
        routes.append(curedge)
        print('%s -> ' %curedge, end=' ')
        done = False

        cnt=0
        while not done:     
            block = True
            #cnt = 0
            while block: 
                if curedge ==destination:
                    break
                curedge = env.get_RoadID(veh) 
                state = env.get_state(veh, curedge) 
                state = np.reshape(state,[1,state_size]) 

                if trained:
                    qvalue, action = agent.get_trainedaction(state)
                    # print('err1 dqnTrainedAgent Qvalue: {} / Action: {}'.format(qvalue,action))
                else:
                    action = agent.get_action(state) 

                nextedge = env.get_nextedge(curedge, action) 
                if nextedge!="" : break
   
            print('%s -> ' %nextedge, end=' ')
            # if nextedge in badpoints: isSucess=False
            routes.append(nextedge)
            
            next_state = env.get_nextstate(veh, nextedge)  
            next_state = np.reshape(state,[1,state_size]) 
            reward, done = env.step(curedge, nextedge) #changeTarget to nextedge
            score += reward

            if not trained: agent.append_sample(state, action, reward, next_state, done)

            if not trained and len(agent.memory)>= agent.train_start:
                agent.train_model()
            # if len(routes) > 30:
            #     done = True
            if score <-700: 
                done = True

            if not trained and done: 
                agent.update_target_model()      #check this section*****************
                simu_time.append(env.sumo.simulation.getTime())
                env.sumoclose()
             
            

            # if done: #check it as well **********************************
            # #     if nextedge=="E15":
            # #         print("Arrived!")
            # #     else:
            # #         isSucess = False 
            # #         print("Nothing")
            # #     break

        
                score_avg = 0.9*score_avg +0.1*score if score_avg!=0 else score
                # score_avg = score
                print("\n****episode : {} | score_avg : {} | memory_length : {} | epsilon : {}".format(episode, score_avg,len(agent.memory), agent.epsilon) )
                
                #1) Reward
                scores.append(-score_avg) #Mean Travel Time
                episodes.append(episode)
                if plotResult: plot_result(num_seed, episodes, scores, dirResult, num_episode)
                              
                '''
                #2) Travel Time(Time Step)
                tree = elemTree.parse(fcdoutput)       
                timestep = tree.find('./timestep[last()]') 
                timestep = float(timestep.get('time'))
                print('Total Time Step: ',timestep)
                travel_times.append(float(timestep))

                pylab.plot(episodes, travel_times, 'b')
                pylab.xlabel('episode')
                pylab.ylabel('Travel Time')
                pylab.savefig('./RL/result/dqn_traveltimes'+str(num_episode)+'.png') 
                '''
                break
                
            
            if trained and done: 
                simu_time.append(env.sumo.simulation.getTime())
                env.sumoclose()
                #mean avg 
                score_avg = 0.9*score_avg +0.1*score if score_avg!=0 else score
                print("\n****Trained episode : {} | score_avg : {} ".format(episode, score_avg) )
                
                # Plot
                scores.append(-score)
                episodes.append(episode)
                if plotResult: plot_trainedresult(num_seed, episodes, scores, dirResult, num_episode)

            curedge = nextedge
            cnt+=1
        print(f"Episode: {episode}, Score :{score_avg}")

        # if isSucess: 
        #     if idxSuccess==-1: idxSuccess = episode
        #     cntSuccess+=1
        # else:
        #     cntSuccess=0
        #     idxSuccess=-1
        # lst_cntSuccess.append(cntSuccess)

        #parser = etree.XMLParser(recover=True)
        #etree.fromstring(fcdoutput, parser=parser)

        et=time.time()
        tt=et-st
        i_new=i_new+tt
        episode_time.append(i_new)
        
         
        

        data =[episode,round(score_avg,2),round(episode_time[-1],2),simu_time[-1]]
        save_data(file_path=file_path,data=data)
        # print("Agent : {} Episode : {} len : {} route : {} reward : {}".format(agent.id,episode,len(routes), routes,score) )
        # print(f"Agent : {agent.id} Episode : {episode}, len : {len(routes)} route : {routes}, reward : {score_avg}")    
    # end = time.time()
    # print('Source Code Time: ',end-start)

    #DQN Weights 
    agent.save_weights()
    
    sys.stdout.flush()
    return agent      


if __name__ == "__main__":
    traci,sumolib = checkSumo()

    net = "Net/dqn.net.xml"
    add = "Add/dqn.add.xml"
    det = "Add/dqn.det.xml"
    trip = "Rou/dqn.trip.xml"
    randomrou = "Rou/dqnrandom.rou.xml"
    sumocfg = "dqn.sumocfg"
    dirResult = 'Result/dqn'
    dirModel = 'Avg_Saved_Models/'
    fcdoutput = 'Output/dqn.fcdoutput.xml'
    
    # badpoints=['E2','E4','E7','E9']
    destination = 'E15'
    successend = ["E15"]
    state_size = 94
    action_size = 3
    
    parser = argparse.ArgumentParser(description="Sumo Client Server")
    parser.add_argument("--config",type=str,default = "config.yaml")
    parser.add_argument("--vehicle",type=str)
    args = parser.parse_args()
    config = parseConfig(args)
    cfg_path = os.path.abspath(config['cfg_file'])

    edgelists = get_alledges(net) 
    dict_connection = calculate_connections(edgelists, net)
    dets = generate_lanedetectionfile(net,det) 
    alldets = get_alldets(edgelists)

    OUT_DIR ='New_Final_Testing'
    trained = True
    veh_id = "veh{}".format(args.vehicle)
    file_path = setup_csv(OUT_DIR,veh_id,trained)
    
    if config['use_gui']:
        sumoBinary = sumolib.checkBinary('sumo-gui')
    else:
        sumoBinary = sumolib.checkBinary('sumo')
    num_episode = config['total_episodes']
    #2) Run in DQN environment
    num_seed = random.randrange(1000)
    if trained:
        agent = dqnTrainedAgent(veh_id, num_seed, edgelists, dict_connection, state_size, action_size,num_episode, dirModel)
    else:
        agent = dqnAgent(veh_id, num_seed, edgelists, dict_connection, state_size, action_size,num_episode, dirModel)
   
    num_seed = random.randrange(1000)
    # while True: 
    #     file = dirModel + str(num_episode)+'_'+str(num_seed)+'.h5'
    #     if not os.path.isfile(file): break
    
    dqn_run(agent,num_seed, trained, sumoBinary, config['plot_results'], num_episode, net, trip, randomrou, add, dirResult,dirModel,
    sumocfg, fcdoutput, edgelists,alldets, dict_connection,veh_id,destination,state_size, action_size,config,file_path)
end_time=time.time()
Execution_time=end_time-start_time
print(f"+++++++++ Overall Execution time is==> {Execution_time}+++++++++++")