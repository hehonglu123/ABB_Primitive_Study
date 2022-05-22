from pandas import read_csv, DataFrame
import sys, copy
sys.path.append('../data')
sys.path.append('../toolbox')
from toolbox_circular_fit import *
from abb_motion_program_exec_client import *
from robots_def import *
import matplotlib.pyplot as plt
from lambda_calc import *
from qpsolvers import solve_qp
from scipy.optimize import fminbound


def search_func(alpha,qdot,q_all,curve):
	q_next=q_all+alpha*qdot
	pose_all=curve.flatten()
	for j in range(len(curve)):
		pose_temp=robot.fwd(q_next[6*j:6*(j+1)])
		pose_all[num_joints*j:num_joints*j+3]=pose_temp.p
		pose_all[num_joints*j+3:num_joints*j+6]=pose_temp.R[:,-1]

	return np.sum(np.linalg.norm(pose_all.reshape(len(curve),6)-curve,axis=1))


def barrier1_temp(x):
	a=0.2;b=0;c=-0.05
	return a*np.square(x)+b*x+c
def barrier1(x):
	a=10;b=-1;e=0.5;l=5;
	return -np.divide(a*b*(x-e),l)

def barrier2(x):
	a=10;b=-1;e=0.1;l=5;
	return -np.divide(a*b*(x-e),l)

live=False
if live:
	plt.ion()
	fig = plt.figure(figsize=(12,8))
	ax = fig.add_subplot(111, projection='3d')

robot=abb6640(d=50)
data_set='movel_30_car/'
data_dir='../data/'


curve_js = read_csv(data_dir+data_set+'Curve_js.csv',header=None).values
curve = read_csv(data_dir+data_set+'Curve_in_base_frame.csv',header=None).values

###calculate lam_j
print('lam_j: ',np.sum(np.linalg.norm(np.diff(curve_js,axis=0),axis=1)))

curve=curve[::100]
curve_js=curve_js[::100]

num_joints=len(curve_js[0])
N=len(curve_js)

###get jacobian for all points
Jp_all=[]
JR_all=[]
for i in range(N):
	R_cur=robot.fwd(curve_js[i]).R
	J=robot.jacobian(curve_js[i])
	Jp_all.append(J[3:])
	#modify jacobian here
	JR_all.append(-hat(R_cur[:,-1])@J[:3])
Jp_all=np.array(Jp_all)
JR_all=np.array(JR_all)

D1=np.zeros((num_joints*(N-1),num_joints*N))
D1[:num_joints*(N-1),:num_joints*(N-1)]=np.eye(num_joints*(N-1))
D1[:num_joints*(N-1),num_joints:num_joints*(N)]-=np.eye(num_joints*(N-1))

D2=D1[:-num_joints,:-num_joints]@D1

q_all=curve_js.flatten()
pose_all=curve.flatten()

#qp parameters
H=D1.T@D1 + D2.T@D2
H += 0.001*np.eye(len(H))
lb=-np.ones(num_joints*len(curve_js))
ub=np.ones(num_joints*len(curve_js))
for i in range(100):

	#qp formation	
	f=(H@q_all).T
	###path constraint
	diff=curve-pose_all.reshape(N,num_joints)
	distance_p=np.linalg.norm(diff[:,:3],axis=1)
	distance_R=np.linalg.norm(diff[:,3:],axis=1)

	G1=np.zeros((N,num_joints*N))	#position
	G2=np.zeros((N,num_joints*N))	#normal
	for j in range(N):
		G1[j,6*j:6*(j+1)]=(diff[j][:3]/np.linalg.norm(diff[j][:3]))@Jp_all[j]
		G2[j,6*j:6*(j+1)]=(diff[j][3:]/np.linalg.norm(diff[j][3:]))@JR_all[j]

	h1=barrier1_temp(distance_p)
	h2=barrier2(distance_R)
	h=np.hstack((h1,h2))
	G=np.vstack((G1,G2))

	print(h1[-1])

	if np.linalg.norm(G)==0:
		dq=solve_qp(H,f,lb=0.00001*lb,ub=0.00001*ub)
		alpha=1
		# print(dq)
	else:
		dq=solve_qp(H,f,G=-G,h=-h,lb=0.0001*lb,ub=0.0001*ub)
		# alpha=fminbound(search_func,0,1,args=(dq,q_all,curve,))
		alpha=1
		# print(alpha)

	#update curve
	q_all+=alpha*dq
	###calculate lam_j
	# print('lam_j: ',np.sum(np.linalg.norm(np.diff(q_all.reshape((N,num_joints)),axis=0),axis=1)))

	#verify constraint
	# print('act',(diff[-1,:3]/np.linalg.norm(diff[-1,:3]))@Jp_all[-1]@dq[-6:],h1[-1])
	print(diff[-1,:3],Jp_all[-1]@dq[-6:])

	Jp_all=[]
	JR_all=[]

	for j in range(N):
		q_cur=q_all[6*j:6*(j+1)]
		pose_temp=robot.fwd(q_cur)
		pose_all[num_joints*j:num_joints*j+3]=pose_temp.p
		pose_all[num_joints*j+3:num_joints*j+6]=pose_temp.R[:,-1]

		J=robot.jacobian(q_cur)
		Jp_all.append(J[3:])
		#modify jacobian here
		JR_all.append(-hat(pose_temp.R[:,-1])@J[:3])
		
	Jp_all=np.array(Jp_all)
	JR_all=np.array(JR_all)

	if live:
		plt.pause(0.001)
		ax = plt.axes(projection='3d')
		ax.plot3D(curve[:,0], curve[:,1],curve[:,2], 'red',label='original')
		ax.plot3D(pose_all[0::6], pose_all[1::6],pose_all[2::6], 'gray',label='output')
		plt.legend()
		plt.draw()

plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(curve[:,0], curve[:,1],curve[:,2], 'red',label='original')
ax.plot3D(pose_all[0::6], pose_all[1::6],pose_all[2::6], 'gray',label='output')
plt.legend()
plt.show()

DataFrame(q_all.reshape((len(curve),num_joints))).to_csv('output/Curve_js_qp2.csv',header=False,index=False)

