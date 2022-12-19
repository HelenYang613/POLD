import yaml
import pandas as pd
import numpy as np

def y2c(path):
    with open(path,"r") as f:
        coor = yaml.load(f.read(), Loader = yaml.Loader)

    space = np.zeros([len(coor),8])
    for i in range(len(coor)):
        temp = list(coor[i].values())
        space[i,0] = temp[1][0][0]
        space[i,1] = temp[1][0][1]
        space[i,2] = temp[1][1][0]
        space[i,3] = temp[1][1][1]
        space[i,4] = temp[1][2][0]
        space[i,5] = temp[1][2][1]
        space[i,6] = temp[1][3][0]
        space[i,7] = temp[1][3][1]
    return space