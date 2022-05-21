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



def barrier1(x):
	a=10;b=-1;e=0.5;l=5;
	return -np.divide(a*b*(x-e),l)

def barrier2(x):
	a=10;b=-1;e=0.1;l=5;
	return -np.divide(a*b*(x-e),l)

def calc_relative_path(curve_js):
	###convert curve to initial tool frame
	pose_init=robot.fwd(curve_js[0])
	R_init=pose_init.R
	p_init=pose_init.p
	curve_new=[]
	curve_normal_new=[]
	Jp_all_new=[]
	JR_all_new=[]
	curve_in_base_frame=[]
	for i in range(len(curve_js)):
		pose=robot.fwd(curve_js[i])
		curve_new.append(R_init.T@(pose.p-p_init))
		curve_normal_new.append(R_init.T@pose.R[:,-1])
		J=robot.jacobian(curve_js[i])
		Jp_all_new.append(R_init.T@J[3:])
		JR_all_new.append(R_init.T@(-hat(pose.R[:,-1])@J[:3]))
		curve_in_base_frame.append(pose.p)
	############################################################init relative to last#################################
	pose_last=robot.fwd(curve_js[-1])
	R_last=pose_last.R
	p_last=pose_last.p
	p_init_last=pose_last.R.T@(p_init-p_last)
	norm_init_last=pose_last.R.T@R_init[:,-1]
	J_init=robot.jacobian(curve_js[0])
	Jp_init_last=R_last.T@J_init[3:]
	JR_init_last=R_last.T@(-hat(R_init[:,-1])@J_init[:3])

	return np.array(curve_new),np.array(curve_normal_new),Jp_all_new,JR_all_new,np.array(curve_in_base_frame),p_init_last,norm_init_last,Jp_init_last,JR_init_last


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

curve=curve[::100]
curve_js=curve_js[::100]

#convert curve to initial tool frame
curve_new,curve_normal_new,Jp_all,JR_all,_,p_init_last_default,norm_init_last_default,Jp_init_last,JR_init_last=calc_relative_path(curve_js)
curve_relative=np.hstack((curve_new,curve_normal_new))

num_joints=len(curve_js[0])
N=len(curve_js)

#form difference matrix
D1=np.zeros((num_joints*(N-1),num_joints*N))
D1[:num_joints*(N-1),:num_joints*(N-1)]=np.eye(num_joints*(N-1))
D1[:num_joints*(N-1),num_joints:num_joints*(N)]-=np.eye(num_joints*(N-1))

D2=D1[:-num_joints,:-num_joints]@D1

#copy initial trajectory configs vectors
q_all=copy.deepcopy(curve_js).flatten()
pose_all=copy.deepcopy(curve_relative).flatten()
p_init_last=p_init_last_default
norm_init_last=norm_init_last_default

#qp parameters
H=D1.T@D1 + D2.T@D2
H += 0.001*np.eye(len(H))
lb=-np.ones(num_joints*len(curve_js))
ub=np.ones(num_joints*len(curve_js))
for i in range(500):

	#qp formation	
	f=(H@q_all).T
	###path constraint
	diff=curve_relative-pose_all.reshape(N,num_joints)
	distance_p=np.linalg.norm(diff[:,:3],axis=1)
	distance_R=np.linalg.norm(diff[:,3:],axis=1)

	G1=np.zeros((N,num_joints*N))	#position
	G2=np.zeros((N,num_joints*N))	#normal
	for j in range(1,N):
		G1[j,6*j:6*(j+1)]=(diff[j][:3]/np.linalg.norm(diff[j][:3]))@Jp_all[j]
		G2[j,6*j:6*(j+1)]=(diff[j][3:]/np.linalg.norm(diff[j][3:]))@JR_all[j]
	###form constraint for initial point differently
	diff_init_p=p_init_last_default-p_init_last
	diff_init_norm=norm_init_last_default-norm_init_last
	G1[0,0:6]=(diff_init_p/np.linalg.norm(diff_init_p))@Jp_init_last
	G2[0,0:6]=(diff_init_norm/np.linalg.norm(diff_init_norm))@JR_init_last

	###concatenate constraints
	h1=barrier1(np.hstack((diff_init_p,distance_p[3:])))
	h2=barrier2(np.hstack((diff_init_norm,distance_R[3:])))
	h=np.hstack((h1,h2))
	G=np.vstack((G1,G2))

	# print(h1[-1])

	if np.linalg.norm(diff)==0:
		dq=solve_qp(H,f,lb=0.00001*lb,ub=0.00001*ub)
		alpha=1

	else:
		dq=solve_qp(H,f,G=-G,h=-h)
		# alpha=fminbound(search_func,0,1,args=(dq,q_all,curve,))
		alpha=0.001
		# print(alpha)


	#update curve
	q_all+=alpha*dq

	###calculate lam_j
	# print('lam_j: ',np.sum(np.linalg.norm(np.diff(q_all.reshape((N,num_joints)),axis=0),axis=1)))

	#verify constraint
	# print('act',(diff[-1,:3]/np.linalg.norm(diff[-1,:3]))@Jp_all[-1]@dq[-6:],h1[-1])

	curve_new,curve_normal_new,Jp_all,JR_all,curve_in_base_frame,p_init_last,norm_init_last,Jp_init_last,JR_init_last=calc_relative_path(q_all.reshape((N,num_joints)))
	pose_all=np.hstack((curve_new,curve_normal_new)).flatten()

	if live:
		plt.pause(0.001)
		ax = plt.axes(projection='3d')
		ax.plot3D(curve[:,0], curve[:,1],curve[:,2], 'red',label='original')
		ax.plot3D(curve_in_base_frame[:,0], curve_in_base_frame[:,1],curve_in_base_frame[:,2], 'gray',label='output')
		plt.legend()
		plt.draw()

plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(curve[:,0], curve[:,1],curve[:,2], 'red',label='original')
ax.plot3D(curve_in_base_frame[:,0], curve_in_base_frame[:,1],curve_in_base_frame[:,2], 'gray',label='output')
plt.legend()
plt.show()

DataFrame(q_all.reshape((len(curve),num_joints))).to_csv('output/Curve_js_qp2.csv',header=False,index=False)

