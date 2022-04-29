import numpy as np
import time, sys
import matplotlib.pyplot as plt
from pandas import *
from general_robotics_toolbox import *
from toolbox_circular_fit import *
sys.path.append('../../toolbox')
from utils import *
from robots_def import *
from exe_toolbox import *
from lambda_calc import *

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

# dataset='movec+movel_smooth'

robot=abb6640(d=50)
###generate a continuous arc, with linear orientation
start_p = np.array([2376.26152,	1089.256029,	746.5836202])
mid_p = np.array([2367.209475,	993.3095092,	706.2353064])
end_p = np.array([2350.540522,	893.5152332,	679.5172614])

c,r=circle_from_3point(start_p,end_p,mid_p)
arc=arc_from_3point(start_p,end_p,mid_p,N=2500)
lam=calc_lam_cs(arc)
avg_distance=np.average(np.gradient(lam))
N_plane=np.cross(c-start_p,c-end_p)
l_vec=np.cross(c-end_p,N_plane)
l_vec=l_vec/np.linalg.norm(l_vec)
line=end_p+np.outer(lam,l_vec)

curve=np.vstack((arc,line))

#get orientation only
q_init=np.array([0.626837286,	0.839988113,	-0.245742828,	1.700793354,	-0.899330476,	0.768529957])
q_end=np.array([0.575135423,	0.921567943,	-0.147742612,	1.569344818,	-1.375486144,	0.605301197])
R_init=robot.fwd(q_init).R
R_end=robot.fwd(q_end).R
#interpolate orientation and solve inv kin
curve_js=[q_init]
R_all=[R_init]
k,theta=R2rot(np.dot(R_end,R_init.T))
for i in range(1,len(curve)):
	angle=theta*i/(len(curve)-1)
	R_temp=rot(k,angle)
	R_all.append(np.dot(R_temp,R_init))
	q_all=np.array(robot.inv(curve[i],R_all[-1]))
	###choose inv_kin closest to previous joints
	temp_q=q_all-curve_js[-1]
	order=np.argsort(np.linalg.norm(temp_q,axis=1))
	curve_js.append(q_all[order[0]])

curve_js=np.array(curve_js)
R_all=np.array(R_all)

visualize(curve,R_all[:,:,-1])
###print commanded points
idx=np.linspace(0,len(arc)-1,3).astype(int)
print(np.degrees(q_init))
for i in idx:
	if i==0:
		continue
	print(arc[i])
	print(R2q(R_all[i]))
	print(quadrant(curve_js[i]))
print(curve[-1])
print(R2q(R_all[-1]))
print(quadrant(curve_js[-1]))
# ###########save to csv####################
df=DataFrame({'breakpoints':np.array([0,2500,5000]),'primitives':['movej_fit','movec_fit','movel_fit'],'points':[[q_init],[mid_p,end_p],[curve[-1]]]})
df.to_csv('command.csv',header=True,index=False)

df=DataFrame({'x':curve[:,0],'y':curve[:,1], 'z':curve[:,2],'x_dir':R_all[:,0,-1],'y_dir':R_all[:,1,-1], 'z_dir':R_all[:,2,-1]})
df.to_csv('Curve_in_base_frame.csv',header=False,index=False)
DataFrame(curve_js).to_csv('Curve_js.csv',header=False,index=False)