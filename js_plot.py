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
curve_js = read_csv('data/'+dataset+'/Curve_js.csv',header=None).values
curve_js_ljl = read_csv('data/moveljl_30_car/Curve_js.csv',header=None).values
curve_js_lcl = read_csv('data/movelcl_30_car2/Curve_js.csv',header=None).values


###get breakpoints location
data = read_csv('data/'+dataset+'/command.csv')
breakpoints=np.array(data['breakpoints'].tolist())
breakpoints[1:]=breakpoints[1:]-1


data_dir='execution/'+dataset+'/'
speed={'v500':v500}
zone={'z10':z10}

for s in speed:
	for z in zone:
		###read in recorded joint data
		data = read_csv(data_dir+"curve_exe_"+s+'_'+z+".csv")
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

		lam1=calc_lam_js(curve_js,robot)
		lam2=calc_lam_js(curve_exe_js,robot)


		for i in range(6):
			plt.figure(i)
			plt.plot(lam1,curve_js[:,i],label='original')
			plt.plot(lam2,curve_exe_js[:,i],label='execution')
			plt.plot(lam1,curve_js_lcl[:,i],label='Circular')
			plt.legend()

			plt.title('J'+str(i+1)+' '+s+' '+z)
		plt.show()
		