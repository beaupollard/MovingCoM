import time
from zmqRemoteApi import RemoteAPIClient
import numpy as np
import math
import copy
import numpy as np
import json
import matplotlib.pyplot as plt
import torch
import RL_net
import random
from collections import deque, namedtuple

def optimize_model():
    if len(memory) < batch_size:
        return
    loss_out=0

    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                        batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state).reshape((batch_size,-1))
    action_batch = torch.cat(batch.action).reshape((batch_size,1))
    reward_batch = torch.cat(batch.reward)#.reshape((batch_size,1))

    state_action_values = policy_net.forward(state_batch).gather(1, action_batch)
    # state_actions = torch.cat([state_action_values[i,action_batch[i]] for i in range(batch_size)])

    next_state_values = torch.zeros(batch_size)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(torch.reshape(non_final_next_states,(-1,len(state_batch[0])))).max(1)[0]

    expected_state_action_values = (next_state_values * gamma) + reward_batch
    # Compute Huber loss
    criterion = torch.nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    loss_out+=loss.item()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    return loss_out

def steering(sim,body_id,motor_ids,radius,velo):
    
    # sim.setObjectOrientation(body_id,-1,[0.,0.,3.14/4])
    eulerAngles=sim.getObjectOrientation(body_id,-1)

    for i,motor in enumerate(motor_ids[:2]):
        if i % 2 == 0:
            motor_direction=-1
        else:
            motor_direction=1
        rot_velo=(velo/60/radius[i])+motor_direction*eulerAngles[-1]
        sim.setJointTargetVelocity(motor,rot_velo)

def end_sim(sim,final_pos,body_id):
    pin=sim.getObjectPosition(body_id,-1)
    if pin[0]<final_pos[0] and pin[2]>final_pos[1]:
        return True, pin
    else:
        return False, pin

def torque_rec(sim,motor_ids,torque):
    tor=[]
    for i in motor_ids:
        tor.append(sim.getJointForce(i))
    return torque.append(tor)

def set_radius(nodes):
    radius=[]
    prev_track=False
    for i in nodes:
        if i["name"]=='prop':
            if prev_track==True:
                prev_track=False
            else:
                radius.append(i['radius'])
                radius.append(i['radius'])
                if i['type']=='track':
                    prev_track=True  
    return radius 

def get_state():
    vel, omega = sim.getObjectVelocity(body_ids)
    pose=sim.getObjectPose(body_ids,-1)
    state=(np.array(vel))
    state=np.concatenate((state,np.array(omega)))
    state=np.concatenate((state,np.array(pose)))
    return torch.tensor(np.concatenate((np.array(state).flatten(),np.array([sim.getJointPosition(motor_ids[-1]),sim.getJointVelocity(motor_ids[-1])]))),dtype=torch.float)

def set_action(state):
    global steps_done

    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end)*math.exp(-1.*steps_done/eps_decay)
    steps_done+=1
    action = policy_net(state)

    with torch.no_grad():
        action = policy_net(state).argmax().item()

    sim.setJointTargetPosition(motor_ids[-1],possible_actions[action])
    # sim.setJointTargetForce(motor_ids[-1],possible_actions[action])
    return torch.tensor([action],dtype=torch.int64)   

def main_run(motor_ids,body_id,sim,final_pos):
    
    radius=[4.5/39.37,4.5/39.37]
    velo=25. 
    torque=[]
    client.setStepping(False)
    client.setStepping(True)

    sim.startSimulation()

    end_sim_var=False
    count=0
    
    ## Run simulation ##
    while end_sim_var==False:
        client.step()
        current_state = get_state()
        steering(sim,body_id,motor_ids,radius,velo)
        current_action = set_action(current_state)
        client.step()
        observation = get_state()

        if sim.getSimulationTime()>30. or current_state[10]>0.5:
            time=sim.getSimulationTime()
            end_sim_var=True
            reward = 0
            next_state=None     
        
        count+=1        
    sim.stopSimulation()

    while sim.getSimulationState()!=sim.simulation_stopped:
        pass
    
    return current_state[6].item(), current_state[8].item()

def get_objects():
    motor_ids=[]
    ids=['/back_right','/back_left','/payload']
    for i in ids:
        motor_ids.append(sim.getObject(i))
    
    body_id=sim.getObject('/mainbody')

    return motor_ids, body_id

class memory_class():
    def __init__(self,capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

memory=memory_class(10000)
policy_net=RL_net.RL_NN()
target_net=RL_net.RL_NN()
target_net.load_state_dict(torch.load("./target_net"))
policy_net.load_state_dict(torch.load("./policy_net"))
# target_net.load_state_dict(policy_net.state_dict())

possible_actions=policy_net.action_inputs

# client = RemoteAPIClient(port=23000)
client = RemoteAPIClient(port=23000)
sim = client.getObject('sim')
motor_ids, body_ids = get_objects()

eps_start = 0.9
eps_end = 0.05
eps_decay = 1000
batch_size=128
tau = 0.005
gamma = 0.9
LR=1e-3
loss_out=[]
optimizer = torch.optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
xpos_rec=[]
zpos_rec=[]
steps_done=0
run_nums=0

# change_track_links()
while run_nums<10000:
    xpos, zpos= main_run(motor_ids, body_ids, sim, [-6.0, 2.1])
    xpos_rec.append(xpos)
    zpos_rec.append(zpos)
    run_nums+=1
print('stop')