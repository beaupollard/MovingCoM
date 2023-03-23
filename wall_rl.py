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
from vrep_ctrl import sim_run
import itertools

def loss_q(o,a,o2,rw,d):
    q_optimizer.zero_grad()

    q1 = ac.q1(o,a)
    q2 = ac.q2(o,a)

    with torch.no_grad():
        a2, logp_a2 = ac.pi(o2)
            # Target Q-values
        q1_pi_targ = ac_target.q1(o2, a2)
        q2_pi_targ = ac_target.q2(o2, a2)
        q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
        backup = rw + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)
    
    loss_q1 = ((q1 - backup)**2).mean()
    loss_q2 = ((q2 - backup)**2).mean()
    loss_q = loss_q1 + loss_q2
    loss_q.backward()
    # torch.nn.utils.clip_grad_norm_(ac.q_params,10)
    q_optimizer.step()
    return loss_q.item()

def loss_pi(o):
    pi_optimizer.zero_grad()
    pi = ac.pi(o)
    pi, logp_pi = ac.pi(o)
    q1_pi = ac.q1(o, pi)
    q2_pi = ac.q2(o, pi)
    q_pi = torch.min(q1_pi, q2_pi)

    loss_pi=(alpha * logp_pi - q_pi).mean()
    loss_pi.backward()
    torch.nn.utils.clip_grad_norm_(ac.pi.parameters(),6)
    pi_optimizer.step()

    return loss_pi.item()

def set_action(state):
    global steps_done

    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end)*math.exp(-1.*steps_done/eps_decay)
    steps_done+=1
    action = ac.act(state)
    if sample>eps_threshold:
        with torch.no_grad():
            action = ac.act(state)
    else:
        for i in range(len(action)):
            action[i] = np.random.normal(0,6.)

    return torch.tensor([action],dtype=torch.float)

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
    next_state_batch = torch.cat(batch.next_state).reshape((batch_size,-1))
    action_batch = torch.cat(batch.action).reshape((batch_size,-1))
    reward_batch = torch.cat(batch.reward)#.reshape((batch_size,1))
    done_batch = torch.cat(batch.done)#.reshape((batch_size,1))

    loss_qout=loss_q(state_batch,action_batch,next_state_batch,reward_batch,done_batch)

    for p in ac.q1.parameters():
        p.requires_grad = False
    for p in ac.q2.parameters():
        p.requires_grad = False

    loss_piout=loss_pi(state_batch)

    for p in ac.q1.parameters():
        p.requires_grad = True
    for p in ac.q2.parameters():
        p.requires_grad = True

    with torch.no_grad():
        for p, p_targ in zip(ac.parameters(), ac_target.parameters()):
            # NB: We use an in-place operations "mul_", "add_" to update target
            # params, as opposed to "mul" and "add", which would make new tensors.
            p_targ.data.mul_(0.99)
            p_targ.data.add_((1 - 0.99) * p.data)
    # print(loss_piout, loss_qout)
    # state_action_values = A_policy_net.forward(state_batch).gather(1, action_batch)

    # next_state_values = torch.zeros(batch_size)
    # with torch.no_grad():
    #     next_state_values[non_final_mask] = target_net(torch.reshape(non_final_next_states,(-1,len(state_batch[0])))).max(1)[0]

    # expected_state_action_values = (next_state_values * gamma) + reward_batch
    # # Compute Huber loss
    # criterion = torch.nn.SmoothL1Loss()
    # loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # # Optimize the model
    # optimizer.zero_grad()
    # loss.backward()
    # loss_out+=loss.item()
    # # In-place gradient clipping
    # torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    # optimizer.step()
    # return loss_qout, loss_piout

def main_run():
    end_sim_var=False
    sim_scene.step_sim(np.zeros(5))
    while end_sim_var==False:

        current_state=torch.tensor(sim_scene.record_state(),dtype=torch.float)

        action=set_action(current_state)
        action2sim=np.array([action[0][0].item(),action[0][0].item(),action[0][1].item(),action[0][1].item(),action[0][2].item()])
        flag, observation, sim_time = sim_scene.step_sim(action2sim)
        observation=torch.tensor(observation,dtype=torch.float)
        if flag==True:
            
            end_sim_var=True
            reward = 0
            next_state=copy.copy(observation)
            done=1
            memory.push(current_state,action,next_state,torch.tensor([reward],dtype=torch.float),torch.tensor([done],dtype=torch.float))
            return sim_time
        else:
            reward=np.array(sim_scene.sim.getObjectPosition(sim_scene.body_ids[0],-1))[0]-init_state[0]+np.array(sim_scene.sim.getObjectVelocity(sim_scene.body_ids[0]))[0][0]#-sim_scene.sim.getSimulationTime()
            next_state=copy.copy(observation)
            done=0
        memory.push(current_state,action,next_state,torch.tensor([reward],dtype=torch.float),torch.tensor([done],dtype=torch.float))
        optimize_model()

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
                        ('state', 'action', 'next_state', 'reward', 'done'))


## Initialize the networks ##
memory=memory_class(10000)
ac=RL_net.MLPActorCritic()
ac.load_state_dict(torch.load('./AC_net_unstruct'))
ac_target=copy.deepcopy(ac)
# Q_policy_net=RL_net.RL_Q()
# Q_target_net=copy.deepcopy(Q_policy_net)


# policy_net.load_state_dict(torch.load("./policy_net"))


eps_start = 0.9
eps_end = 0.05
eps_decay = 100
batch_size=128
tau = 0.005
gamma = 0.9
LR=1e-3
loss_out=[]
# Q_optimizer = torch.optim.AdamW(Q_policy_net.parameters(), lr=LR, amsgrad=True)
# A_optimizer = torch.optim.AdamW(A_policy_net.parameters(), lr=LR, amsgrad=True)
xpos_rec=[]
zpos_rec=[]
time_rec=[]
iters_rec=[]
steps_done=10000
run_nums=0
alpha = 0.2
sim_time_rec=[]

q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())
pi_optimizer = torch.optim.Adam(ac.pi.parameters(), lr=1e-3)
q_optimizer = torch.optim.Adam(q_params, lr=1e-3)

## Initialize the simulations ##
joint_names=['/Wheel_assem1','/Wheel_assem3','/Wheel_assem4','/Wheel_assem2','/Payload_x']
body_names=['/Frame1']
scene_name='/home/beau/Documents/moving_cg/wheel_vehicle_flat_x.ttt'
count=0
# change_track_links()
while run_nums<10000:

    sim_scene=sim_run(joint_names,body_names,scene_name) 
    init_state=np.array(sim_scene.sim.getObjectPosition(sim_scene.body_ids[0],-1))
    sim_time_rec.append(main_run())
    if count==100:
        torch.save(ac.state_dict(), 'AC_net_unstruct2')
        count=0
    else:
        count+=1
    print(run_nums)
    run_nums+=1
print('stop')
torch.save(ac.state_dict(), 'AC_net_unstruct')
# torch.save(target_net.state_dict(), 'target_net_unstruct')