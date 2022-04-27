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
	
	j0=jointtarget([35.915130615234375, 48.12777328491211, -14.080026626586914, 97.44828033447266, -51.52783966064453, 44.03352355957031],[0]*6)
	r1 = robtarget([2363.051025390625, 953.3972778320312, 691.204833984375], [0.288772851228714, 0.656049370765686, 0.42526233196258545, 0.5525951385498047], confdata(0,1,0,1),[0]*6)
	r2 = robtarget([2331.660888671875, 811.3377685546875, 668.3482055664062], [0.47620800137519836, 0.6388378739356995, 0.46708744764328003, 0.3833293616771698], confdata(0,1,0,1),[0]*6)

	r3 = robtarget([2314.811767578125, 751.4532470703125, 666.4553833007812], [0.47620800137519836, 0.6388378739356995, 0.46708744764328003, 0.3833293616771698], confdata(0,1,0,1),[0]*6)
	r4 = robtarget([2296.10009765625, 692.2407836914062, 669.3971557617188], [0.5281842947006226, 0.6120356321334839, 0.4721687436103821, 0.35141199827194214], confdata(0,0,0,1),[0]*6)

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