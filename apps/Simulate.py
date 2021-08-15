# Python packages
import matplotlib.pyplot as plt
from matplotlib.text import Annotation
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from streamlit_metrics import metric, metric_row 
import os
import json
from PIL import Image

# Importing py scripts 
from Plot import make_heatmap 

output_base = 0

def app():
    global output_base, img_path
    image = Image.open('logo_transparant2.png')

    st.write("")

    img_path = os.path.abspath('data/002_test.png')

    @st.cache
    def long_running_import():
        # Get output of calculation
        with open("data.json", 'r') as f:
            output_base = json.load(f)
        return output_base

    def make_annotation(office, workers):
        #all loc
        annotate1 = []
        coord1x = []
        coord1y = []

        annotate2 = []
        coord2x = []
        coord2y = []

        for i in office:
            annotate1.append(i['id'])
            coord1x.append(i['cd_x'])
            coord1y.append(i['cd_y'])
        for i in workers:
            annotate2.append(i['who'])
            coord2x.append(i['cd_x'])
            coord2y.append(i['cd_y'])

        return annotate1+annotate2, coord1x+coord2x, coord1y+coord2y

    #interactive plots
    def video_slider(t, what, cap):
        global img_path
        global output_base

        #the dataframe in the json
        data = pd.DataFrame.from_dict(json.loads(output_base[str(cap)]))
        #simulation column value at this capacity
        sim = data["Simulation_output"].values[0] 
        sim_output = np.array(sim)

        #time_scale for the slider
        time_scale = [x[0] for x in sim]

        #simulation objects
        office, workers = data["Office"].values[0], data["Workers"].values[0]
        annotate ,coordx, coordy = make_annotation(office, workers)
        
        #heatmap locations
        new_df = make_heatmap(sim_output=sim)
        
        #base image (background)
        img = plt.imread(img_path)
        fig, axs = plt.subplots(figsize=(10,10))
        plt.title(f't = {t/4}h')

        axs.imshow(np.flipud(img), cmap='gray', origin='lower')
        axs.set_xlim([0, img.shape[1]])
        axs.set_ylim([0, img.shape[0]])
        
        #dropdown menu selection #1
        if what == "Paths":
            #plot all objects in this simulation variant
            for i in range(len(annotate)):
                axs.annotate(annotate[i], (coordx[i], coordy[i]),va="bottom", ha="center",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"), size=10)

            #plotting walking paths
            for j in sim_output[np.array(time_scale)==t]:
                axs.plot([j[1]['cd_x'], j[2]['cd_x']], [j[1]['cd_y'], j[2]['cd_y']], 'o-', color='black')
            st.pyplot(fig) 

        #dropdown menu selection #2
        if what == 'Heatmap':
            #plotting the heatmap by first selecting the points to kde closest to eachother 
            if len(new_df.iloc[np.argmin(abs(new_df.index-float(t))),0]) <= 2:
                x_plot = []
                y_plot = []
            else:
                x_plot, y_plot = new_df.iloc[np.argmin(abs(new_df.index-float(t))),0], new_df.iloc[np.argmin(abs(new_df.index-float(t))),1]

            sns.kdeplot(x=x_plot, y=y_plot, shade=True, shade_lowest=False, alpha=0.5,
                    cmap='coolwarm', ax = axs) 
            st.pyplot(fig) 
        return None

    #data
    output_base = long_running_import()
    
    #Sidebar
    st.sidebar.write("# Variables")
    what = st.sidebar.selectbox("Analyse", ['Paths', 'Heatmap'])

    st.sidebar.write("## Capacity")
    cap = st.sidebar.slider("Number of Employees", min_value=1, max_value=len(output_base))

    st.sidebar.write("## Time")
    t = st.sidebar.slider("Timestep in 15 minutes", min_value=24, max_value=72)

    #site layout
    st.write("""
    # Indoor Simulation Model Omnia

    Simulate office behaviour with just a floorplan. 
    This is the MVP of our simulation model. Here we can see a office building. 

    Overview Metrics:
    """)

    data = pd.DataFrame.from_dict(json.loads(output_base[str(cap)]))

    metric_row(
        {
            "Capacity": cap,
            "Meters walked": int(sum(data["Activity_timeseries_y"].values[0])/100),
            "Contact per Person": int(data["Hygiene_score"].values[0]),
        }
    )

    video_slider(t, what, cap) 

    agree = st.checkbox('More')
    if agree:
        fig2, ax = plt.subplots()
        ax.set_title("Variations")
        for i in range(len(data) ):
            x = np.array(data["Activity_timeseries_x"].values[i])*4
            y = np.array(data["Activity_timeseries_y"].values[i])/100
            plt.plot(x,y, alpha=0.6) 
        ax.set_xlabel("Time in [15 minutes]")
        ax.set_ylabel("Activity")
        st.pyplot(fig2)
