import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sc
import numpy as np
import pandas as pd
import random
import json
import operator

import base64
import os
import re
import cv2

def make_heatmap(sim_output):
    """
    Function takes in output file from the simulation and creates activity points from which KDE can be done in visualisation functions
    Made for sns.kdeplot() and x, y are the only inputs which are presented in a DataFrame called df
    #possible KDE via scipy or sklearn for griddata to modify better.
    """
    when_simulated = np.array([i[0] for i in sim_output])
    x_coord, y_coord = [], []
    for i in when_simulated:
        #filter on time to get accumulated heatmap
        sim_filter_time = np.array(sim_output)[when_simulated<=i]
        sim_filtered = sim_filter_time.tolist()
        
        #x_b = [i[1]['cd_x'] for i in sim_filtered]
        #y_b = [i[1]['cd_y'] for i in sim_filtered]

        x_e = [i[2]['cd_x'] for i in sim_filtered]
        y_e = [i[2]['cd_y'] for i in sim_filtered]

        x_coord.append(x_e)
        y_coord.append(y_e)

    #make dataframe with list
    df = pd.DataFrame(list(zip(x_coord,y_coord)), columns=['x','y'], index = when_simulated)
    
    #remove duplicates from time stamps
    df = df[~df.index.duplicated(keep='first')]
    return df