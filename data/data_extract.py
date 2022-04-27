import numpy as np
import time, sys
import matplotlib.pyplot as plt
from pandas import *
from general_robotics_toolbox import *
sys.path.append('../toolbox')
from utils import *


curve_all = read_csv('movec/curve_fit.csv').values
curve=curve_all[:,:3]

DataFrame(curve_all[:,[0,1,2,5,8,11]]).to_csv('movec/Curve_in_base_frame.csv',header=False,index=False)