from toolbox_circular_fit import *

def movec_insertion(curve,p1,p2,R1,R2,slope1,slope2):

	arc,circle=circle_fit_w_2slope(curve,p1,p2,slope1,slope2)

	k,theta=R2rot(np.dot(R2,R1.T))
	R_arc=[]
	for i in range(1,len(arc)-1):
		angle=theta*i/(len(arc)-1)
		R_temp=rot(k,angle)
		R_arc.append(np.dot(R_temp,R1))

	return np.array(arc[1:-1]),np.array(R_arc)


def movej_insertion(q1,q2):
	return