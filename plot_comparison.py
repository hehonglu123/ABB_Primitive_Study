import numpy as np
from general_robotics_toolbox import *
from pandas import *
import sys
# sys.path.append('../abb_motion_program_exec')
from abb_motion_program_exec_client import *
sys.path.append('toolbox')
from robots_def import *
from error_check import *
from lambda_calc import *
from utils import *


robot=abb6640(d=50)
dataset='movel_30_car'
###read in original curve
curve = read_csv('data/'+dataset+'/Curve_in_base_frame.csv',header=None).values
curve_js = read_csv('data/'+dataset+'/Curve_js.csv',header=None).values

###get breakpoints location
data = read_csv('data/'+dataset+'/command.csv')
breakpoints=np.array(data['breakpoints'].tolist())
breakpoints[1:]=breakpoints[1:]-1

data=read_csv('execution/movel_30_car/curve_exe_v500_z10.csv')
q1=data[' J1'].tolist()
q2=data[' J2'].tolist()
q3=data[' J3'].tolist()
q4=data[' J4'].tolist()
q5=data[' J5'].tolist()
q6=data[' J6'].tolist()
cmd_num=np.array(data[' cmd_num'].tolist()).astype(float)
start_idx=np.where(cmd_num==4)[0][0]
curve_exe_js=np.radians(np.vstack((q1,q2,q3,q4,q5,q6)).T.astype(float)[start_idx:])
timestamp=np.array(data['timestamp'].tolist()[start_idx:]).astype(float)
timestep=np.average(timestamp[1:]-timestamp[:-1])

###get cartesian info
curve_exe1=[]
curve_exe_R1=[]
act_speed=[]

for i in range(len(curve_exe_js)):
	robot_pose=robot.fwd(curve_exe_js[i])
	curve_exe1.append(robot_pose.p)
	curve_exe_R1.append(robot_pose.R)
	if i>0:
		act_speed.append(np.linalg.norm(curve_exe1[-1]-curve_exe1[-2])/timestep)
curve_exe1=np.array(curve_exe1)
curve_exe_R1=np.array(curve_exe_R1)

data=read_csv('execution/movel_30_car_close/curve_exe_v500_z10.csv')
q1=data[' J1'].tolist()
q2=data[' J2'].tolist()
q3=data[' J3'].tolist()
q4=data[' J4'].tolist()
q5=data[' J5'].tolist()
q6=data[' J6'].tolist()
cmd_num=np.array(data[' cmd_num'].tolist()).astype(float)
start_idx=np.where(cmd_num==4)[0][0]
curve_exe_js=np.radians(np.vstack((q1,q2,q3,q4,q5,q6)).T.astype(float)[start_idx:])
timestamp=np.array(data['timestamp'].tolist()[start_idx:]).astype(float)
timestep=np.average(timestamp[1:]-timestamp[:-1])

###get cartesian info
curve_exe2=[]
curve_exe_R2=[]
act_speed=[]

for i in range(len(curve_exe_js)):
	robot_pose=robot.fwd(curve_exe_js[i])
	curve_exe2.append(robot_pose.p)
	curve_exe_R2.append(robot_pose.R)
	if i>0:
		act_speed.append(np.linalg.norm(curve_exe2[-1]-curve_exe2[-2])/timestep)
curve_exe2=np.array(curve_exe2)
curve_exe_R2=np.array(curve_exe_R2)



###plot original curve
plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(curve_exe1[:,0]-500, curve_exe1[:,1],curve_exe1[:,2], 'red',label='Curve 1')
#plot execution curve
ax.plot3D(curve_exe2[:,0], curve_exe2[:,1],curve_exe2[:,2], 'green',label='Curve 2')
plt.legend()
plt.show()