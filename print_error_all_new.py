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

dataset='movec'

###read in original curve
curve = read_csv('data/'+dataset+'/Curve_in_base_frame.csv',header=None).values
curve_js = read_csv('data/'+dataset+'/Curve_js.csv',header=None).values

###get breakpoints location
data = read_csv('data/'+dataset+'/command.csv')
breakpoints=np.array(data['breakpoints'].tolist())
breakpoints[1:]=breakpoints[1:]-1


data_dir='execution/'+dataset+'/'
speed={'v50':v50,'v100':v100,'v200':v200,'v400':v400}
zone={'z10':z10,'z1':z1}#,'z5':z5,'z1':z1,'fine':fine}
for s in speed:
	for z in zone:
		###read in recorded joint data
		col_names=['timestamp', 'cmd_num', 'J1', 'J2','J3', 'J4', 'J5', 'J6'] 
		data = read_csv(data_dir+"curve_exe_"+s+'_'+z+".csv",names=col_names)
		q1=data['J1'].tolist()[1:]
		q2=data['J2'].tolist()[1:]
		q3=data['J3'].tolist()[1:]
		q4=data['J4'].tolist()[1:]
		q5=data['J5'].tolist()[1:]
		q6=data['J6'].tolist()[1:]
		cmd_num=np.array(data['cmd_num'].tolist()[1:]).astype(float)
		start_idx=np.where(cmd_num==3)[0][0]
		curve_exe_js=np.radians(np.vstack((q1,q2,q3,q4,q5,q6)).T.astype(float)[start_idx:])
		timestamp=np.array(data['timestamp'].tolist()[start_idx:]).astype(float)
		timestep=np.average(timestamp[1:]-timestamp[:-1])

		###get cartesian info
		curve_exe=[]
		curve_exe_R=[]
		act_speed=[]

		for i in range(len(curve_exe_js)):
			robot_pose=robot.fwd(curve_exe_js[i])
			curve_exe.append(robot_pose.p)
			curve_exe_R.append(robot_pose.R)
			if i>0:
				act_speed.append(np.linalg.norm(curve_exe[-1]-curve_exe[-2])/timestep)
		curve_exe=np.array(curve_exe)
		curve_exe_R=np.array(curve_exe_R)
		lam=calc_lam_cs(curve_exe)

		###plot original curve
		plt.figure()
		plt.title(s+' '+z)
		ax = plt.axes(projection='3d')
		ax.plot3D(curve[:,0], curve[:,1],curve[:,2], 'red',label='original')
		ax.scatter3D(curve[breakpoints,0], curve[breakpoints,1],curve[breakpoints,2], 'blue')
		#plot execution curve
		ax.plot3D(curve_exe[:,0], curve_exe[:,1],curve_exe[:,2], 'green',label='execution')
		

		error,angle_error=calc_all_error_w_normal(curve_exe,curve[:,:3],curve_exe_R[:,:,-1],curve[:,3:])
		fig, ax1 = plt.subplots()
		ax2 = ax1.twinx()
		ax1.plot(lam[1:],act_speed, 'g-', label='Speed')
		ax2.plot(lam, error, 'b-',label='Cartesian Error')
		ax2.plot(lam, np.degrees(angle_error), 'y-',label='Normal Error')
		ax1.legend(loc=0)
		ax2.legend(loc=0)
		ax1.set_xlabel('lambda (mm)')
		ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
		ax2.set_ylabel('Error (mm/deg)', color='b')
		ax2.set_ylim(0,1)

		plt.title(dataset+' error vs lambda'+' '+s+' '+z)
		plt.show()
		