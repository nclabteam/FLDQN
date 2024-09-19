import yaml
import os
import sys
import logging
from xml.etree.ElementTree import parse
from collections import defaultdict
import xml.etree.ElementTree as ET
import csv 
import datetime
import random

def setup_csv(out_dir,veh_id,trained):
    out_dir = os.path.abspath(out_dir)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    current_datetime = datetime.datetime.now()
    current_time = current_datetime.time()
    current_time = str(current_datetime).split('.')[0].replace(':','-').replace(' ','_')
    if not trained:
        file_name = f'{current_time}_{veh_id}_train.csv'
    else:
        file_name = f'{current_time}_{veh_id}_test.csv'
    file_path = os.path.join(out_dir,file_name)
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode','Waiting_Time','Total_Time','simu_time'])
    return file_path


def save_data(file_path,data):
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)


def setUpLogger():
    log_level = logging.WARN
    logging.basicConfig(format='%(asctime)s %(filename)s:%(lineno)d %(message)s', level=log_level)
    logger = logging.getLogger(__name__)
    logger.info("Logger started")
    return logger

def parseConfig(args):
    yaml_file = args.config
    with open(file=yaml_file) as file:
        try:
            config = yaml.safe_load(file)   
            p_config = {}
            p_config['veh_id'] = args.vehicle
            p_config['plot_results'] = config['common']['plot_results']
            p_config['cfg_file'] = config['common']['cfg_file']
            p_config['total_episodes'] = config['common']['total_episodes']
            p_config['port'] = config['common']['port']
            p_config['server_ip'] = config['common']['server_ip']
            p_config['num_clients'] = config['sumo']['num_clients']
            p_config['use_gui'] = config['sumo']['use_gui']
            p_config['start'] = config['sumo']['start']
            p_config['start_edges'] = config['sumo']['start_edges']
            p_config['quit_on_end'] = config['sumo']['quit_on_end']
            return p_config
        except yaml.YAMLError as exc:
            print(exc)

def checkSumo():
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'],'tools')
        sys.path.append(tools)
        try:
            import traci
            import sumolib
            return traci, sumolib
        except ImportError:
            raise EnvironmentError("Declare SUMO_HOME environment")
    else:
        sys.exit('Declare environment variable "SUMO_HOME"')

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

def generate_detectionfile(net, det):
    #generate det.xml file by setting a detector at the end of each lane (-10m)
    with open(det,"w") as detections:
        alledges = get_alledges(net)
        alllanes = [edge +'_0' for edge in alledges]
        alldets =  [edge.replace("E","D") for edge in alledges]
        
        pos = -10
        print('<additional>', file = detections)
        for i in range(len(alledges)):
            print(' <e1Detector id="%s" lane="%s" pos="%i" freq="30" file="cross.out" friendlyPos="x"/>'
            %(alldets[i], alllanes[i],pos), file = detections)
        print('</additional>', file = detections)
        return alldets

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


def getRandomRoute(dictconnection,sumo):
    start_edge = 0
    dest_edge = 0
    valid_connections = {}
    # print(dictconnection)
    while ("ON A PAS UNE ROUTE VALIDE"):
        while (start_edge == dest_edge):
            start_edge = random.randint(0, len(dictconnection)-1)
            dest_edge = random.randint(0, len(dictconnection)-1)
        i = 0
        j = 0
        for it in dictconnection:
            if(i == start_edge):
                pos = it
                j += 1
            if(i == dest_edge):
                dest = it
                j += 1
            if(j == 2):
                break
            i += 1
            if(i > len(dictconnection)):
                i = 0
        route = sumo.simulation.findRoute(pos, dest)
        nodes = sumo.simulation.findRoute(pos, dest).edges
        if (len(nodes) > 3):
            print("out")
            break
        else:
            start_edge = 0
            dest_edge = 0
    return nodes


# def plot_result(episodenum, lst_cntSuccess):
#     ax = plt.figure().gca()
#     ax.yaxis.set_major_locator(MaxNLocator(integer=True))
#     plt.xticks(range(episodenum),rotation=45)
#     plt.plot(lst_cntSuccess, 'r-')
    
#     plt.savefig('./Qlearning/Result/qlresult%i.png' %episodenum )