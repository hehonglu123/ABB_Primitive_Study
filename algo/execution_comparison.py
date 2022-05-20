import numpy as np
from general_robotics_toolbox import *
from pandas import *
import sys
# sys.path.append('../abb_motion_program_exec')
from abb_motion_program_exec_client import *
sys.path.append('../toolbox')
from robots_def import *
from error_check import *
from lambda_calc import *
from utils import *


robot=abb6640(d=50)
data_dir='../data/'
dataset='movel_30_car'

###read in original curve
curve = read_csv(data_dir+dataset+'/Curve_in_base_frame.csv',header=None).values
curve_js = read_csv(data_dir+dataset+'/Curve_js.csv',header=None).values

###read in optimized curve
curve_qp_js = read_csv('output/Curve_js_qp2.csv',header=None).values
curve_qp=[]
for i in range(len(curve_qp_js)):
	curve_qp.append(robot.fwd(curve_qp_js[i]).p)
curve_qp=np.array(curve_qp)

lam_qp=calc_lam_cs(curve_qp)
lamdot_qp=calc_lamdot(curve_qp_js,lam_qp,robot,step=1)


###get breakpoints location
data = read_csv(data_dir+dataset+'/command.csv')
breakpoints=np.array(data['breakpoints'].tolist())
breakpoints[1:]=breakpoints[1:]-1


output_dir='../execution/qp_opt/'
speed=['vmax']
zone=['z10']
for s in speed:
	for z in zone:
		###read in recorded joint data
		data=read_csv(output_dir+'curve_exe_'+s+'_'+z+'.csv')
		q1=data[' J1'].tolist()
		q2=data[' J2'].tolist()
		q3=data[' J3'].tolist()
		q4=data[' J4'].tolist()
		q5=data[' J5'].tolist()
		q6=data[' J6'].tolist()
		cmd_num=np.array(data[' cmd_num'].tolist()).astype(float)
		idx = np.absolute(cmd_num-5).argmin()
		start_idx=np.where(cmd_num==cmd_num[idx])[0][0]
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
		ax.plot3D(curve[:,0], curve[:,1],curve[:,2], 'red',label='Original')
		ax.plot3D(curve_qp[:,0], curve_qp[:,1],curve_qp[:,2], 'gray',label='QP Output Curve')
		ax.scatter3D(curve[breakpoints,0], curve[breakpoints,1],curve[breakpoints,2], 'blue')
		#plot execution curve
		ax.plot3D(curve_exe[:,0], curve_exe[:,1],curve_exe[:,2], 'green',label='Execution')
		
		plt.figure()
		plt.plot(lam[1:],act_speed, 'g-', label='Execution Speed')
		plt.plot(lam_qp,lamdot_qp,'r-',label='Lamdot from QP Output')
		plt.xlabel('Path Length-lambda (mm)')
		plt.ylabel('Speed/Lamdadot (mm/s)')
		plt.legend()

		# plt.title(dataset+' error vs lambda'+' '+s+' '+z)
		plt.show()
		