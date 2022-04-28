import numpy as np
import time, sys
import matplotlib.pyplot as plt
from pandas import *
from general_robotics_toolbox import *
sys.path.append('../toolbox')
from utils import *


def gen_curve_normal(seed_vec,curve):
	curve_normal=[]
	error=[]
	for i in range(len(curve)-1):
		moving_direction=curve[i+1]-curve[i]
		moving_direction=moving_direction/np.linalg.norm(moving_direction)
		curve_normal_temp=VectorPlaneProjection(seed_vec,moving_direction)

		curve_normal.append(curve_normal_temp/np.linalg.norm(curve_normal_temp))
		error.append(np.dot(moving_direction,curve_normal[-1]))

	curve_normal.append(curve_normal[-1])
	return np.array(curve_normal)

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


start_p = np.array([1300,1000,600])
mid_p = np.array([2300,0,600])
end_p = np.array([1300,-1000,600])
lam_f=np.linalg.norm(start_p-mid_p)+np.linalg.norm(mid_p-end_p)
lam=np.linspace(0,lam_f,num=50000)


a1,b1,c1=lineFromPoints([lam[0],start_p[0]],[lam[int(len(lam)/2)],mid_p[0]])
a2,b2,c2=lineFromPoints([lam[0],start_p[1]],[lam[int(len(lam)/2)],mid_p[1]])
a3,b3,c3=lineFromPoints([lam[0],start_p[2]],[lam[int(len(lam)/2)],mid_p[2]])
seg1=np.vstack(((-a1*lam[:int(len(lam)/2)]-c1)/b1,(-a2*lam[:int(len(lam)/2)]-c2)/b2,(-a3*lam[:int(len(lam)/2)]-c3)/b3)).T

a1,b1,c1=lineFromPoints([lam[int(len(lam)/2)],mid_p[0]],[lam[-1],end_p[0]])
a2,b2,c2=lineFromPoints([lam[int(len(lam)/2)],mid_p[1]],[lam[-1],end_p[1]])
a3,b3,c3=lineFromPoints([lam[int(len(lam)/2)],mid_p[2]],[lam[-1],end_p[2]])
seg2=np.vstack(((-a1*lam[int(len(lam)/2):]-c1)/b1,(-a2*lam[int(len(lam)/2):]-c2)/b2,(-a3*lam[int(len(lam)/2):]-c3)/b3)).T

curve=np.vstack((seg1,seg2))
curve_normal=gen_curve_normal([0.97324,	0.091,	-0.21101],curve)

visualize(curve,curve_normal)
###########save to csv####################
df=DataFrame({'x':curve[:,0],'y':curve[:,1], 'z':curve[:,2],'x_dir':curve_normal[:,0],'y_dir':curve_normal[:,1], 'z_dir':curve_normal[:,2]})
df.to_csv('movel/Curve_in_base_frame.csv',header=False,index=False)