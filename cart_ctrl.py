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

def end_sim(jstate):

    if abs(jstate)>12.5*math.pi/180:
        return True
    else:
        return False

def get_state():
    vels=[]
    vel, omega = sim.getObjectVelocity(motor_ids[1])
    pos= sim.getObjectPosition(motor_ids[1],-1)
    # for i in motor_ids[:1]:
    #     vel, omega = sim.getObjectVelocity(i)
        
    vels.append(vel[0])
    vels.append(pos[0])
        # vels.append(omega)
    # pose = sim.getObjectPose(i,motor_ids[1])
        


    return np.concatenate((np.array(vels).flatten(),np.array([sim.getJointPosition(motor_ids[-1]),sim.getJointVelocity(motor_ids[-1])])))

def set_action(state):
    global steps_done
    # global prev_speed
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end)*math.exp(-1.*steps_done/eps_decay)
    steps_done+=1
    action = policy_net(state)
    if sample>eps_threshold:
        with torch.no_grad():
            action = policy_net(state).argmax().item()
    else:
        # if state[-2].item()>0.:
        #     action=0
        # else:
        #     action=1
        action = random.randint(0,1)
    # prev_speed+=possible_actions[action]
    # sim.setJointTargetPosition(motor_ids[0],prev_speed)
    # sim.setJointTargetVelocity(motor_ids[0],possible_actions[action])
    sim.setJointTargetForce(motor_ids[0],possible_actions[action])
    return action

def main_run():
    client.setStepping(False)
    client.setStepping(True)
    sim.startSimulation()

    end_sim_var=False
    
    ## Run simulation ##
    while end_sim_var==False:
        

        state=get_state()
        action=set_action(torch.tensor(state,dtype=torch.float))
        client.step()

        if end_sim(state[-2])==True or sim.getSimulationTime()>30.:
            end_sim_var=True
            reward=0
            next_state=None
        else:
            reward=1
            next_state=torch.tensor(get_state(),dtype=torch.float)
        memory.push(torch.tensor(state,dtype=torch.float),torch.tensor([action],dtype=torch.int64),next_state,torch.tensor([reward],dtype=torch.float))
        l2 = optimize_model()
        loss_out.append(l2)

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
        target_net.load_state_dict(target_net_state_dict)
    run_times.append(sim.getSimulationTime())
    sim.stopSimulation()

    while sim.getSimulationState()!=sim.simulation_stopped:
        pass

    return

def get_objects():
    motor_ids=[]
    ids=['/ctrl','/cart','/pole_joint']
    for i in ids:
        motor_ids.append(sim.getObject(i))
    

    return motor_ids

class memory_class():
    def __init__(self,capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def optimize_model():
    if len(memory) < batch_size:
        return
    loss_out=0
    for j in range(10):
        transitions = memory.sample(batch_size)
        batch = Transition(*zip(*transitions))
        # non_final_mask=[]
        # for i in (batch.next_state):
        #     if i is not None:
        #         non_final_mask.append(True)
        #     else:
        #         non_final_mask.append(False)
        # non_final_mask=torch.tensor(non_final_mask)
        # xx=lambda s: print(s)
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

def plot_results(ave_window=100):
    ave_times=np.zeros(len(run_times))
    for i in range(ave_window,len(run_times)):
        ave_times[i]=(sum(run_times[i-ave_window:i])/ave_window)
    plt.plot(run_times)
    plt.plot(ave_times)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

memory=memory_class(10000)
policy_net=RL_net.RL_NN()
target_net=RL_net.RL_NN()
target_net.load_state_dict(policy_net.state_dict())

possible_actions=policy_net.action_inputs

client = RemoteAPIClient()
sim = client.getObject('sim')
client.setStepping(True)
motor_ids = get_objects()

eps_start = 0.9
eps_end = 0.05
eps_decay = 1000
batch_size=128
tau = 0.005
gamma = 0.9
LR=1e-3
loss_out=[]
optimizer = torch.optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

run_times=[]
steps_done=0
run_nums=0

while run_nums<10000:
    prev_speed=0
    sim.setJointPosition(motor_ids[-1],random.uniform(-0.05,0.05))
    # client = RemoteAPIClient()
    # sim = client.getObject('sim')   
    # client.setStepping(True)
    main_run()
    run_nums+=1
print('done')