import numpy as np
from pandas import *
import sys, traceback, time, copy
from general_robotics_toolbox import *
import matplotlib.pyplot as plt
from qpsolvers import solve_qp
sys.path.append('../toolbox')
from robots_def import *
from utils import *
from lambda_calc import *
from error_check import *

def single_arm_stepwise_optimize(robot,q_init,lam,lamdot_des,curve,curve_normal):

		q_all=[q_init]
		lam_out=[0]
		curve_out=[curve[0]]
		curve_normal_out=[curve_normal[0]]
		ts=0.01
		total_time=lam[-1]/lamdot_des
		total_timestep=int(total_time/ts)
		idx=0
		lam_qp=0
		K_trak=100
		qdot_prev=[]
		act_speed=[]
		# for i in range(1,total_timestep):
		while lam_qp<lam[-1] and idx<len(curve)-1:
			#find corresponding idx in curve & curve normal
			prev_idx=np.abs(lam-lam_qp).argmin()
			idx=np.abs(lam-(lam_qp+ts*lamdot_des)).argmin()

			pose_now=robot.fwd(q_all[-1])
			
			Kw=10
			Kq=.01*np.eye(6)    #small value to make sure positive definite
			KR=np.eye(3)        #gains for position and orientation error

			J=robot.jacobian(q_all[-1])        #calculate current Jacobian
			Jp=J[3:,:]
			JR=J[:3,:]
			JR_mod=-np.dot(hat(pose_now.R[:,-1]),JR)

			H=np.dot(np.transpose(Jp),Jp)+Kq+Kw*np.dot(np.transpose(JR_mod),JR_mod)
			H=(H+np.transpose(H))/2
			#####vd = - v_des - K ( x - x_des)
			v_des=(curve[idx]-curve[prev_idx])/ts
			vd=v_des+K_trak*(curve[prev_idx]-pose_now.p)
			ezdot_des=(curve_normal[idx]-curve_normal[prev_idx])/ts
			ezdotd=ezdot_des+K_trak*(curve_normal[prev_idx]-pose_now.R[:,-1])

			#####vd = K ( x_des_next - x_cur)
			# vd=(curve[idx]-pose_now.p)/ts
			# vd=lamdot_des*vd/np.linalg.norm(vd)
			# ezdotd=(curve_normal[idx]-pose_now.R[:,-1])/ts

			print(np.linalg.norm(vd))

			f=-np.dot(np.transpose(Jp),vd)-Kw*np.dot(np.transpose(JR_mod),ezdotd)
			if len(qdot_prev)==0:
				lb=-robot.joint_vel_limit
				ub=robot.joint_vel_limit
			else:
				# lb=np.maximum(-robot.joint_vel_limit,(qdot_prev-robot.joint_acc_limit)*ts)
				# ub=np.minimum(robot.joint_vel_limit,(qdot_prev+robot.joint_acc_limit)*ts)
				lb=-robot.joint_vel_limit
				ub=robot.joint_vel_limit

			qdot=solve_qp(H,f,lb=lb,ub=ub)
			qdot_prev=qdot

			q_all.append(q_all[-1]+ts*qdot)

			curve_out.append(pose_now.p)
			curve_normal_out.append(pose_now.R[:,-1])

			
			lam_qp+=np.linalg.norm(robot.fwd(q_all[-1]).p-pose_now.p)
			lam_out.append(lam_qp)

			act_speed.append(np.linalg.norm(np.dot(Jp,qdot)))

		q_all=np.array(q_all)
		curve_out=np.array(curve_out)
		curve_normal_out=np.array(curve_normal_out)

		return q_all,lam_out,curve_out,curve_normal_out,act_speed

def main():

	robot=abb6640(d=50)

	###read in points
	data_set='movel_30_car/'
	data_dir='../data/'
	zone=1

	data = read_csv(data_dir+data_set+'command.csv')
	breakpoints=np.array(data['breakpoints'].tolist())
	act_breakpoints=copy.deepcopy(breakpoints)
	act_breakpoints[1:]=act_breakpoints[1:]-1

	curve_js = read_csv(data_dir+data_set+'Curve_js.csv',header=None).values
	curve = read_csv(data_dir+data_set+'Curve_in_base_frame.csv',header=None).values



	lam=calc_lam_cs(curve)
	lamdot_des=500

	q_all,lam_out,curve_out,curve_normal_out,act_speed=single_arm_stepwise_optimize(robot,curve_js[0],lam,lamdot_des,curve[:,:3],curve[:,3:])

	###plot results
	plt.figure(0)
	ax = plt.axes(projection='3d')
	ax.plot3D(curve[:,0], curve[:,1], curve[:,2], c='red')
	ax.plot3D(curve_out[:,0], curve_out[:,1], curve_out[:,2], c='green')
	# plt.show()

	print(calc_max_error_w_normal(curve_out[2:],curve[:,:3],curve_normal_out[2:],curve[:,3:]))
	# df=DataFrame({'j1':q_all[:,0],'j2':q_all[:,1],'j3':q_all[:,2],'j4':q_all[:,3],'j5':q_all[:,4],'j6':q_all[:,5]})
	# df.to_csv('curve_qp_js.csv',header=False,index=False)

	lamdot=calc_lamdot(q_all,lam_out,robot,1)

	plt.figure(1)
	plt.plot(lam_out[1:],act_speed)
	plt.title("TCP speed (jacobian) vs lambda, v_des= "+str(lamdot_des))
	plt.ylabel('speed (mm/s)')
	plt.xlabel('lambda (mm)')


	plt.figure(2)
	plt.plot(lam_out,lamdot, label='Lambda Dot')
	plt.title("lamdot vs lambda, v_des= "+str(lamdot_des))
	plt.ylabel('lamdot (mm/s)')
	plt.xlabel('lambda (mm)')
	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()