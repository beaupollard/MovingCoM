from zmqRemoteApi import RemoteAPIClient
import numpy as np
import math
import copy

class sim_run():
    def __init__(self,joint_names,body_names,scene_name,stepping=True,wall_dims=[3.,9.,0.4]):
        self.client = RemoteAPIClient(port=23000)
        self.sim = self.client.getObject('sim')        
        self.sim.loadScene(scene_name)
        self.client.setStepping(stepping)

        self.joint_ids=self.get_obj(joint_names)
        # self.sim.setJointPosition(self.joint_ids[-1],-90*math.pi/180+np.random.normal(0,0.02))
        self.body_ids=self.get_obj(body_names)
        self.init_pose=np.array(self.sim.getObjectPose(self.body_ids[0],-1))

        # Create the wall
        self.wall_dims=wall_dims
        self.body_ids.append(self.sim.createPrimitiveShape(self.sim.primitiveshape_cuboid,wall_dims))
        self.sim.setObjectPosition(self.body_ids[-1],-1,[wall_dims[0]/2-0.5,0.,wall_dims[-1]/2])
        self.sim.setObjectInt32Param(self.body_ids[-1],self.sim.shapeintparam_respondable,1)
        self.final_pos=[3.0,0.75]
        self.sim.startSimulation()

    def get_obj(self,names):
        ids=[]
        for i in names:
            ids.append(self.sim.getObject(i))
        return ids
    
    def set_joint_velos(self,action):
        for i, act in enumerate(action):
            self.sim.setJointTargetVelocity(self.joint_ids[i],act)
    
    def record_state(self):
        # Location of the wall
        wall_loc=np.array(self.sim.getObjectPosition(self.body_ids[-1],-1))
        wall_loc[-1]=2*wall_loc[-1]
        wall_loc[0]=wall_loc[0]-self.wall_dims[0]/2
        
        # State of the vehicle
        vel, omega = self.sim.getObjectVelocity(self.body_ids[0])
        pose=np.array(self.sim.getObjectPose(self.body_ids[0],-1))
        pose[:3]=pose[:3]-self.init_pose[:3]-wall_loc
        joints=[self.sim.getJointPosition(self.joint_ids[-1])]
        for i in self.joint_ids:
            joints.append(self.sim.getJointVelocity(i))
        
        state=copy.copy(pose)
        state=np.concatenate((state,np.array(omega)))
        state=np.concatenate((state,np.array(vel)))
        state=np.concatenate((state,np.array(joints)))
        return state

    def step_sim(self,action):
        self.set_joint_velos(action)
        self.client.step()
        state=self.record_state()
        flag, sim_time=self.end_sim()
        return flag, state, sim_time
    
    def end_sim(self):
        # if abs(self.sim.getJointPosition(self.joint_ids[-1])+math.pi/2)*180/math.pi>15 or self.sim.getSimulationTime()>10:
        #     sim_time=self.sim.getSimulationTime()
        #     self.kill_sim()
        #     return True, sim_time
        # else:
        #     return False, 0
        pin=self.sim.getObjectPosition(self.body_ids[0],-1)
        pin_ori=np.array(self.sim.getObjectOrientation(self.body_ids[0],-1))
        sim_time=self.sim.getSimulationTime()
        pin_max=max(abs(pin_ori))
        if pin[0]>self.final_pos[0]:
            
            self.kill_sim()
            return True, sim_time
        else:
            if pin_max*180/math.pi>90 or self.sim.getSimulationTime()>5.:
                self.kill_sim()
                return True, sim_time
            else:
                return False, sim_time
            
    def kill_sim(self):
        self.sim.stopSimulation()

        while self.sim.getSimulationState()!=self.sim.simulation_stopped:
            pass   

        self.sim.closeScene()
            