import numpy as np
import time, sys
import matplotlib.pyplot as plt
from pandas import *
from general_robotics_toolbox import *
from toolbox_circular_fit import *
sys.path.append('../toolbox')
from utils import *
from robots_def import *
from exe_toolbox import *

def visualize(curve,curve_normal):
	curve=curve[::500]
	curve_normal=curve_normal[::500]
	X, Y, Z = zip(*curve)
	U, V, W = zip(*curve_normal*50)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.quiver(X, Y, Z, U, V, W)
	ax.set_box_aspect((np.ptp(X), np.ptp(Y), np.ptp(Z)))

	plt.show()

robot=abb6640(d=50)
###generate a continuous arc, with linear orientation
start_p = np.array([2376.26152,	1089.256029,	746.5836202])
mid_p = np.array([2331.66100427,  811.33777642,  668.34821038])
end_p = np.array([2296.10018593,  692.24076104,  669.3971284])

arc=arc_from_3point(start_p,end_p,mid_p,N=5000)

#get orientation
q_init=np.array([0.626837286,	0.839988113,	-0.245742828,	1.700793354,	-0.899330476,	0.768529957])
q_end=np.array([0.575135423,	0.921567943,	-0.147742612,	1.569344818,	-1.375486144,	0.605301197])
R_init=robot.fwd(q_init).R
R_end=robot.fwd(q_end).R
#interpolate orientation and solve inv kin
curve_js=[q_init]
R_all=[R_init]
k,theta=R2rot(np.dot(R_end,R_init.T))
for i in range(1,len(arc)):
	angle=theta*i/(len(arc)-1)
	R_temp=rot(k,angle)
	R_all.append(np.dot(R_temp,R_init))
	q_all=np.array(robot.inv(arc[i],R_all[-1]))
	###choose inv_kin closest to previous joints
	temp_q=q_all-curve_js[-1]
	order=np.argsort(np.linalg.norm(temp_q,axis=1))
	curve_js.append(q_all[order[0]])

curve_js=np.array(curve_js)
R_all=np.array(R_all)

visualize(arc,R_all[:,:,-1])
###print commanded points
idx=np.linspace(0,len(arc)-1,5).astype(int)
print(np.degrees(q_init))
for i in idx:
	if i==0:
		continue
	print(arc[i])
	print(R2q(R_all[i]))
	print(quadrant(curve_js[i]))
###########save to csv####################
df=DataFrame({'x':arc[:,0],'y':arc[:,1], 'z':arc[:,2],'x_dir':R_all[:,0,-1],'y_dir':R_all[:,1,-1], 'z_dir':R_all[:,2,-1]})
df.to_csv('movec_smooth/Curve_in_base_frame.csv',header=False,index=False)
DataFrame(curve_js).to_csv('movec_smooth/Curve_js.csv',header=False,index=False)