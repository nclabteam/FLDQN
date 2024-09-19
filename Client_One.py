from typing import Dict, Tuple
import flwr as fl
import tensorflow as tf
import numpy as np
from flwr.common import NDArray, NDArrays, Scalar
from dqnagent import dqnAgent
from dqnrun import dqn_run 
import sumolib
from utils import *
import argparse
import time 

def get_edgesinfo(net):
    tree = parse(net)
    root = tree.getroot()
    alledgelists = root.findall("edge")
    edgesinfo = [x.find("lane").attrib for x in alledgelists]
    return edgesinfo

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
start_time=time.time()
class flowerClient(fl.client.NumPyClient):
    def __init__(self,agent,trained,sumoBinary, plotResult, num_episode, net, trip, randomrou, add, dirResult, dirModel,
                sumocfg, fcdoutput, edgelists, alldets, dict_connection, veh, destination, state_size, action_size, dqn_config, num_seed,file_path) -> None:
        self.agent = agent
        self.trained=trained
        self.num_seed=num_seed
        self.sumoBinary=sumoBinary
        self.plotResult=plotResult
        self.num_episode=num_episode
        self.net=net
        self.trip=trip
        self.randomrou=randomrou
        self.add=add
        self.dirResult=dirResult
        self.dirModel=dirModel
        self.sumocfg=sumocfg
        self.fcdoutput=fcdoutput
        self.edgelists=edgelists
        self.alldets=alldets
        self.dict_connection=dict_connection
        self.veh=veh
        self.destination=destination
        # self.badpoints=badpoints
        self.state_size=state_size
        self.action_size=action_size
        self.dqn_config=dqn_config
        self.file_path = file_path

    def get_parameters(self,config) -> NDArray:
        # Load the model weights from the DQN agent
        weights = self.agent.model.get_weights()
        return weights

    def fit(self, parameters: NDArray,config) -> Tuple[NDArray, Scalar]:
        # Set the model weights received from the server
        self.agent.model.set_weights(parameters)
        # Call your dqn_run function here or integrate its logic into the fit method
        # Pass the necessary arguments to dqn_run method based on your implementation
        print(f"Current round : { config['round']}")
      
        self.agent = dqn_run(self.agent, self.num_seed,self.trained,  self.sumoBinary,  self.plotResult,  self.num_episode, self.net,  self.trip,  self.randomrou, self.add, self.dirResult, self.dirModel,
                 self.sumocfg,  self.fcdoutput,  self.edgelists,  self.alldets, self.dict_connection,  self.veh, self.destination,self.state_size,  self.action_size,  self.dqn_config,self.file_path,config['round'])
       
        self.agent.save_weights()
        
        # Return the updated model weights after training
        weights = self.agent.model.get_weights()
        loss = 0.0  # You can calculate the loss and return it here if needed
        return weights,1000,{}

    def evaluate(self, parameters: NDArray,config) -> Tuple[Scalar, Scalar]:
        # Set the model weights received from the server
        # self.agent.model.set_weights(parameters)

        # Implement the evaluation logic here
        # Return evaluation metrics such as accuracy or any other relevant metrics
        accuracy = 0.0
        loss = 0.0
        return loss, 1000, {"accuracy": float(accuracy)}

# Create an instance of the flowerClients class and start the Flower client
if __name__ == "__main__":
    # num_seed = random.randrange(1000)
    num_seed = 36
    trained = False  # Assuming you want to train the agent
    # sumoBinary = "path/to/sumo/binary"
    plotResult = False  # Change to True if you want to plot the results
    num_episode = 100  # Number of episodes to run
    net = "/home/nclab3/Desktop/sumo/venv-sumo/Net/dqn.net.xml"
    trip = "/home/nclab3/Desktop/sumo/venv-sumo/Rou/dqn.trip.xml"
    det = "Add/dqn.det.xml"
    randomrou = "Rou/dqnrandom.rou.xml"
    add = "/home/nclab3/Desktop/sumo/venv-sumo/Add/dqn.add.xml"
    dirResult = 'Result/dqn'
    dirModel = '/home/nclab3/Desktop/sumo/venv-sumo/Avg_Saved_Models/'
    sumocfg = "/home/nclab3/Desktop/sumo/venv-sumo/dqn.sumocfg"
    fcdoutput = 'Output/dqn.fcdoutput.xml'
    veh = "veh0"  # Vehicle ID
    destination = "E15"  # Destination edge
    # badpoints=['E2','E4','E7','E9']
    state_size = 94  # Size of state vector
    action_size = 3  # Number of possible actions
    config = {}  # Configuration options
    traci,sumolib = checkSumo()
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
    
    OUT_DIR ='Very_Final'
    veh_id = "veh{}".format(args.vehicle)
    file_path = setup_csv(OUT_DIR,veh_id,trained)
    if config['use_gui']:
        sumoBinary = sumolib.checkBinary('sumo-gui')
    else:
        sumoBinary = sumolib.checkBinary('sumo')
    num_episode = config['total_episodes']
    

    num_episode = config['total_episodes']
    #2) Run in DQN environment
    
    agent = dqnAgent(veh_id, num_seed, edgelists, dict_connection, state_size, action_size,num_episode, dirModel)

    client = flowerClient(agent=agent,trained=trained,sumoBinary=sumoBinary,plotResult=plotResult,num_episode=num_episode,net=net,trip=trip,randomrou=randomrou,add=add,
                          dirResult=dirResult,dirModel=dirModel,sumocfg=sumocfg,fcdoutput=fcdoutput,edgelists=edgelists,alldets=alldets,dict_connection=dict_connection,
                          veh=veh_id,destination=destination,state_size=state_size,action_size=action_size,dqn_config=config, num_seed=num_seed,file_path = file_path)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)

end_time=time.time()
Total_time=end_time-start_time
print(f"+++++++++++++++++++++ overall time==>> {Total_time}+++++++++++++++++++++")


 
    
