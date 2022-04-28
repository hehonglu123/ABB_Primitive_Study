from pandas import read_csv
from abb_motion_program_exec_client import *
from general_robotics_toolbox import *
import sys
sys.path.append('../../toolbox')

from exe_toolbox import *
from robots_def import *

def main():
	quatR = R2q(rot([0,1,0],math.radians(30)))
	tool = tooldata(True,pose([75,0,493.30127019],[quatR[0],quatR[1],quatR[2],quatR[3]]),loaddata(1,[0,0,0.001],[1,0,0,0],0,0,0))
	robot=abb6640(d=50)


	v700 = speeddata(700,500,5000,1000)
	speed={'vmax':vmax,'v50':v50,'v100':v100,'v200':v200,'v400':v400}
	zone={'z10':z10,'z5':z5,'z1':z1,'fine':fine}
	
	j0=jointtarget([35.91513093,  48.12777372, -14.08002689,  97.44828101, -51.52784066,  44.03352297],[0]*6)
	r1 = robtarget([2367.19913977,  993.23084106,  706.20834764], [0.35237284, 0.6512594,  0.44131165, 0.50689111], confdata(0,1,0,1),[0]*6)
	r2 = robtarget([2350.52423453,  893.43454485,  679.50156483], [0.41376069, 0.64228317, 0.45453966, 0.45789531], confdata(0,1,0,1),[0]*6)

	r3 = robtarget([2326.63595521,  792.30598085,  667.13349695], [0.47249029, 0.62918053, 0.46484743, 0.40595771], confdata(0,1,0,1),[0]*6)
	r4 = robtarget([2296.10018593,  692.24076104,  669.3971284 ], [0.52818432, 0.61203564, 0.47216874, 0.35141198], confdata(0,0,0,1),[0]*6)

	for s in speed:
		for z in zone:
			mp = MotionProgram(tool=tool)
			mp.MoveAbsJ(j0,v500,fine)
			mp.WaitTime(0.5)
			mp.MoveAbsJ(j0,v50,fine)
			mp.MoveC(r1,r2,speed[s],zone[z])
			mp.MoveC(r3,r4,speed[s],fine)


			print(mp.get_program_rapid())

			client = MotionProgramExecClient()
			log_results = client.execute_motion_program(mp)

			# Write log csv to file
			with open("curve_exe_"+s+'_'+z+".csv","wb") as f:
			   f.write(log_results)

if __name__ == "__main__":
	main()