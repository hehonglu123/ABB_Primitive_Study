########
# This module utilized https://github.com/johnwason/abb_motion_program_exec
# and send whatever the motion primitives that algorithms generate
# to RobotStudio
########

import numpy as np
from general_robotics_toolbox import *
from pandas import read_csv
import sys
# sys.path.append('../abb_motion_program_exec')
from abb_motion_program_exec_client import *
sys.path.append('../toolbox')
from robots_def import *
from MotionSend import *

def main():
    ms = MotionSend()
    # datasets=['movec_smooth','movec_30_car','movec_30_ori','movec+movel_smooth']#,'movel_smooth','movel_30_car','movel_30_ori']
    # datasets=['movel_smooth','movel_30_car','movel_30_ori']
    datasets=['movel_30_car']
    # datasets=['movec_30_car','movec_30_ori','movec+movel_smooth']
    vmax = speeddata(10000,9999999,9999999,999999)
    speed={'v50':v50,'v300':v300,'v500':v500,'v1000':v1000,'v1500':v1500,'vmax':vmax}
    z500 = zonedata(False,500,300,300,30,300,30)

    zone={'z200':z200,'z100':z100,'z50':z50}

    for dataset in datasets:
        for s in speed:
            for z in zone: 
                curve_exe_js=ms.exe_from_file('../data/'+dataset+"/command.csv",'../data/'+dataset+"/Curve_js.csv",speed[s],zone[z])
       

                f = open(dataset+"/curve_exe"+"_"+s+"_"+z+".csv", "w")
                f.write(curve_exe_js)
                f.close()

if __name__ == "__main__":
    main()