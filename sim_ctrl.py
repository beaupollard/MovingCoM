import time
from zmqRemoteApi import RemoteAPIClient
import numpy as np
import math
import copy
import numpy as np
import json
import matplotlib.pyplot as plt

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

def ctrl_cg(motor_id,body_id,filt_omega):
    _, omega = sim.getObjectVelocity(body_id)    
    pose=sim.getObjectPose(body_id,-1)
    # theta=2*math.asin(pose[4])
    # if filt_omega<0:
    #     target_pos=0.2
    # else:
    #     target_pos=-0.3
    target_pos=-0.15
    sim.setJointTargetPosition(motor_id,target_pos)
    # sim.setJointTargetVelocity(motor_id,-omega[1]/5)
    return pose, omega, sim.getJointPosition(motor_id)

def main_run(motor_ids,body_id,sim,final_pos):
    
    radius=[4.5/39.37,4.5/39.37]
    velo=25. 
    torque=[]
    client.setStepping(False)
    client.setStepping(True)

    sim.startSimulation()

    end_sim_var=False
    count=0
    
    pose=[]
    omega=[]
    cg_pos=[]
    filt_omega=0
    ## Run simulation ##
    while end_sim_var==False:
        client.step()
        steering(sim,body_id,motor_ids,radius,velo)
        if len(omega)>3:
            filt_omega=omega[-2][1]+omega[-1][1]+omega[-3][1]
        p2, o2, cg2=ctrl_cg(motor_ids[-1],body_id,filt_omega)
        pose.append(p2)
        omega.append(o2)
        cg_pos.append(cg2)
        # omega=np.array(omega)
        # pose=np.array(pose)
        # torque_rec(sim,motor_ids,torque)
        # success, pin = end_sim(sim,final_pos,body_id)
        if sim.getSimulationTime()>30.:
            time=sim.getSimulationTime()
            end_sim_var=True
        count+=1        
    sim.stopSimulation()
    #     sim.simxSynchronousTrigger(sim_scene.clientID)
    #     sim_scene.steering()
    #     sim_scene.torque_rec()
    #     success=sim_scene.end_sim(final_pos,count*0.05)
    #     if success==True or count*0.05>30.:
    #         end_sim_var=True
    #     count+=1
    # err0=sim.simxStopSimulation(sim_scene.clientID,sim.simx_opmode_oneshot)
    while sim.getSimulationState()!=sim.simulation_stopped:
        pass
    #     # print("Simulation not ending")
    # err1=sim.simxFinish(sim_scene.clientID) # Connect to CoppeliaSim
    # # print(err0,err1)
    pose=np.array(pose)
    omega=np.array(omega)
    cg_pos=np.array(cg_pos)    
    sum_torque=[sum(abs(np.array(torque)[i,:])) for i in range(len(torque))]
    return success, time, sum(sum_torque)/len(sum_torque), max(sum_torque), pin

def get_objects():
    motor_ids=[]
    ids=['/back_right','/back_left','/payload']
    for i in ids:
        motor_ids.append(sim.getObject(i))
    
    body_id=sim.getObject('/mainbody')

    return motor_ids, body_id

# client = RemoteAPIClient(port=23000)
client = RemoteAPIClient(port=23000)
sim = client.getObject('sim')
motor_ids, body_ids = get_objects()
# change_track_links()
main_run(motor_ids, body_ids, sim, [-6.38, 2.2])
