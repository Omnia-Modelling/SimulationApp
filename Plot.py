import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sc
import numpy as np
import pandas as pd
import random
import json
import operator
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import base64
import os
import re
import cv2

# Plotting Functions
def plot_nodes(img_path, office, workers):
    """
    Plotting a graph with a overview off the locations of all nodes.
    #plot words in image inplaats van legend
    """
    img = plt.imread(img_path)
    fig, ax = plt.subplots()
    ax.imshow(np.flipud(img), cmap='gray', origin='lower')
    for i in office:
        ax.scatter(i['cd_x'], i['cd_y'], marker='o', s=50, color='black')
        ax.annotate(i['id'], (i['cd_x'], i['cd_y']),va="bottom", ha="center",
                  bbox=dict(boxstyle="round", fc="w"),
                  arrowprops=dict(arrowstyle="->"), size=8)
    for i in workers:
        ax.scatter(i['cd_x'], i['cd_y'], marker='o', s=50, color='black')
        ax.annotate(i['who'], (i['cd_x'], i['cd_y']),va="bottom", ha="center",
                  bbox=dict(boxstyle="round", fc="w"),
                  arrowprops=dict(arrowstyle="->"), size=8)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

def plot_H_score(l):
    """
    plotting def to plot the H_score for arbitrary amount of nodes over the time
    debug
    """
    for i in l:
        try:
            if i['kind'] in ['toilet', 'lunchroom', 'meeting_room', 'printer', 'coffee_machine']: # all locations
                plt.plot(np.arange(1, 25, 0.25), np.cumsum(i['H_time']), marker='o', label=i['id'])
            else:
                plt.plot(np.arange(1, 25, 0.25), np.cumsum(i['H_time']), marker='o', label=i['who'])
        except:
            plt.plot(np.arange(1, 25, 0.25), np.cumsum(i['H_time']), marker='o', label=i['who'])
            
    plt.title('H_score_day')
    plt.xlabel('Time(Hours)')
    plt.ylabel('H_score')
    plt.legend()
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

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

## Visualisation of Simulation (MP4)
def make_images(df, dir_name='images', img_name='image.png'):
    if not os.path.exists(dir_name):
        os.mkdir("images")
    
    image_filename = img_name
    plotly_logo = imagem_tunel = base64.b64encode(open(image_filename, 'rb').read())

    for i in range(len(new_df)):
        if new_df.iloc[i,:].values[0].all() != 0:
            fig = go.Figure(data= [go.Heatmap(z=new_df.iloc[i,:].values[0], 
                                              showscale=False, 
                                              connectgaps=True, 
                                              zsmooth='best', 
                                            colorscale = 'Hot',
                                            reversescale=True,
                                  opacity=0.55)]
                           )

            fig.update_layout(
                        images= [dict(
                            source='data:image/png;base64,{}'.format(plotly_logo.decode()),
                            xref="paper", yref="paper",
                            x=0, y=1,
                            sizex=1, sizey=1,
                            xanchor="left",
                            yanchor="top",
                            sizing="stretch",
                            opacity=1,
                            layer="below")])
            fig.write_image(f'images/fig{i}.png', scale=6)

def make_video(image_folder = 'images', video_name = 'video.avi'):    
    temp = re.compile("([a-zA-Z]+)([0-9]+)") 
    lista = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images = sorted(lista, key=lambda x: int(temp.match(x).groups()[1]))
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()