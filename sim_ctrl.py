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

def get_state(zmap,xmap):
    vel, omega = sim.getObjectVelocity(body_ids)
    pose=sim.getObjectPose(body_ids,-1)
    outpts=get_height_measures(zmap,xmap,pose)
    state=(np.array(vel))
    state=np.concatenate((state,np.array(omega)))
    state=np.concatenate((state,np.array(pose)))
    state=np.concatenate((np.concatenate((np.array(state).flatten(),np.array([sim.getJointPosition(motor_ids[-2]),sim.getJointVelocity(motor_ids[-2]),sim.getJointPosition(motor_ids[-1]),sim.getJointVelocity(motor_ids[-1])]))),outpts))
    
    return torch.tensor(state,dtype=torch.float)

def set_action(state):
    global steps_done

    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end)*math.exp(-1.*steps_done/eps_decay)
    steps_done+=1
    action = policy_net(state)
    if sample>eps_threshold:
        with torch.no_grad():
            action = policy_net(state).argmax().item()
    else:
        action = random.randint(0,3)

    sim.setJointTargetPosition(motor_ids[-2],possible_actions[action,0])
    sim.setJointTargetPosition(motor_ids[-1],possible_actions[action,1])
    # sim.setJointTargetForce(motor_ids[-1],possible_actions[action])
    return torch.tensor([action],dtype=torch.int64)

def get_height_measures(zmap,xmap,pose):
    for i in range(len(xmap)-1):
        if (pose[0]-xmap[i])*(pose[0]-xmap[i+1])<=0.:
            ind_x=i
        if (pose[1]-xmap[i])*(pose[1]-xmap[i+1])<=0.:
            ind_y=i
    try:
        xpts=xmap[ind_x-8:ind_x-3]
        ypts=xmap[ind_y-6:ind_y+6]
        zpts=zmap[ind_y-6:ind_y+6,ind_x-8:ind_x-3]
        outpts=np.concatenate((np.concatenate((xpts,ypts)),zpts.flatten()))
    except:
        outpts=np.zeros(77)
    
    return outpts
    # for i in range(len(xpts)):
    #     for j in range(len(ypts)):
    #         c0=sim.createPrimitiveShape(sim.primitiveshape_spheroid,[0.1,0.1,0.1])
    #         sim.setObjectPosition(c0,-1,[xpts[i],ypts[j],zpts[j,i]+0.05])            
    # for i in range(6):
    #     for j in range(15):
    #         c0=sim.createPrimitiveShape(sim.primitiveshape_spheroid,[0.1,0.1,0.1])
    #         sim.setObjectPosition(c0,-1,[xmap[ind_x-13-i],xmap[ind_y-5+j],zmap[ind_y-5+j,ind_x-13-i]+0.05])
    # print('hey')

def main_run(motor_ids,body_id,sim,final_pos,zmap,xmap):
    
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
        current_state = get_state(zmap,xmap)
        steering(sim,body_id,motor_ids,radius,velo)
        current_action = set_action(current_state)
        client.step()
        observation = get_state(zmap,xmap)

        if sim.getSimulationTime()>30. or current_state[10]>0.5:
            time=sim.getSimulationTime()
            end_sim_var=True
            reward = 0
            next_state=None
        # elif current_state[6].item()<final_pos[0] and current_state[8].item()>final_pos[1]:
        #     time=sim.getSimulationTime()
        #     end_sim_var=True
        #     reward = 0
        #     next_state=None
        else:
            reward=abs(current_state[6].item()-x_init)+(current_state[8].item())
            next_state=copy.copy(observation)
        memory.push(current_state,current_action,next_state,torch.tensor([reward],dtype=torch.float))

        _ = optimize_model()
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
        target_net.load_state_dict(target_net_state_dict)        
        
        count+=1        
    sim.stopSimulation()

    while sim.getSimulationState()!=sim.simulation_stopped:
        pass
    
    return current_state[6].item(), current_state[8].item(), sim.getSimulationTime(), count

def get_objects():
    motor_ids=[]
    ids=['/back_right','/back_left','/payload','/payload_x']
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

def generate_terrain():
    xfield_len=100
    yfield_len=100
    xlength=15.
    z=[]
    x=[]
    y=[]
    xloc=np.linspace(0,xlength,xfield_len)-xlength/2
    # periody=random.uniform(0.1,2)
    # periodx=random.uniform(0.1,2)
    # height=random.uniform(0.025,0.075)
    slope=random.uniform(10,20)*math.pi/180
    dz=0
    offset_y=70
    for i in range(xfield_len):
        dz=offset_y*xlength/xfield_len*math.tan(slope)
        for j in range(yfield_len):
            if j<offset_y:
                dz-=xlength/xfield_len*math.tan(slope)
            z.append(dz+random.uniform(-0.08,0.08))


            # z.append(dz+height*(math.sin(periodx*i)+math.sin(periody*j))+random.uniform(-0.025,0.025))
    return sim.createHeightfieldShape(2,30.,xfield_len,yfield_len,xlength,z), np.reshape(np.array(z),(xfield_len,yfield_len)), xloc

def create_dots(pts):
    for i in 10:
        sim.createPrimitiveShape()

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

memory=memory_class(10000)
policy_net=RL_net.RL_NN()
target_net=RL_net.RL_NN()
# target_net.load_state_dict(torch.load("./target_net"))
# policy_net.load_state_dict(torch.load("./policy_net"))
# target_net.load_state_dict(policy_net.state_dict())

possible_actions=policy_net.action_inputs

# client = RemoteAPIClient(port=23000)
client = RemoteAPIClient(port=23000)
sim = client.getObject('sim')
sim.loadScene('/home/beau/Documents/moving_cg/U0_tracked_x_y_cg.ttt')
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
time_rec=[]
iters_rec=[]
steps_done=0
run_nums=0

# change_track_links()
while run_nums<10000:
    pose=sim.getObjectPose(body_ids,-1)
    x_init=pose[0]
    terrain_id, zmap, xmap = generate_terrain()
    xpos, zpos, times, iter= main_run(motor_ids, body_ids, sim, [-6.0, 2.1],zmap,xmap)
    time_rec.append(times)
    iters_rec.append(iter)
    xpos_rec.append(xpos)
    zpos_rec.append(zpos)
    run_nums+=1
    sim.removeObject(terrain_id)
    print(run_nums)
print('stop')
torch.save(policy_net.state_dict(), 'policy_net_unstruct')
torch.save(target_net.state_dict(), 'target_net_unstruct')