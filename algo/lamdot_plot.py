from pandas import read_csv, DataFrame
import sys, copy
sys.path.append('../data')
sys.path.append('../toolbox')
from toolbox_circular_fit import *
from robots_def import *
from lambda_calc import *


robot=abb6640(d=50)

data_set='movel_30_car/'
data_dir='../data/'

curve_js = read_csv(data_dir+data_set+'Curve_js.csv',header=None).values
curve = read_csv(data_dir+data_set+'Curve_in_base_frame.csv',header=None).values

lam=calc_lam_cs(curve)

lamdot=calc_lamdot(curve_js,lam,robot,step=1)

curve_js_qp1=read_csv('output/Curve_js_qp1.csv',header=None).values
lam_qp1=calc_lam_js(curve_js_qp1,robot)
lamdot_qp1=calc_lamdot(curve_js_qp1,lam_qp1,robot,step=1)

curve_js_qp2=read_csv('output/Curve_js_qp2.csv',header=None).values
lam_qp2=calc_lam_js(curve_js_qp2,robot)
lamdot_qp2=calc_lamdot(curve_js_qp2,lam_qp2,robot,step=1)

plt.figure()
plt.plot(lam,lamdot,label='originial')
plt.plot(lam_qp1,lamdot_qp1,label='qp1')
plt.plot(lam_qp2,lamdot_qp2,label='qp2')
plt.legend()
plt.xlabel('lambda, path length (mm)')
plt.ylabel('lambda_dot (mm/s)')
plt.show()