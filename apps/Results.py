# Python packages
import matplotlib.pyplot as plt
from matplotlib.text import Annotation
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from streamlit_metrics import metric, metric_row 
import datetime
from zipfile import ZipFile
import os
import json
from PIL import Image

# Importing py scripts 
from apps import heatmap

output_base = 0

def pageOne(sesh):
    global output_base, img_path
    image = Image.open('data/logo_transparant2.png')

    st.write("")

    img_path = os.path.abspath('data/002_test.png')

    #functions
    @st.cache
    def Agent_Simulation_V1():
        d = None  
        data = None  
        with ZipFile("data/data.zip", "r") as z:
            for filename in z.namelist():   
                with z.open(filename) as f:  
                    data = f.read()  
                    d = json.loads(data.decode("utf-8")) 
        return d

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
    def video_slider(t, what, cap, time):
        global img_path
        global output_base

        #the dataframe in the json
        data = pd.DataFrame.from_dict(json.loads(output_base[str(cap)]))

        #simulation column value at this capacity
        dist = data["Contact_score"]/data["Contact_score"].max()
        choice = int(np.argmin(abs(dist.values-0.5)))

        #simulation column value at this capacity
        sim = data["Simulation_output"].values[choice] 
        sim_output = np.array(sim)

        #time_scale for the slider
        time_scale = [x[0] for x in sim]

        #simulation objects
        office, workers = data["Office"].values[choice], data["Workers"].values[choice]
        annotate ,coordx, coordy = make_annotation(office, workers)
        
        #heatmap locations
        new_df = heatmap.make_heatmap(sim_output=sim)
        
        #base image (background)
        img = plt.imread(img_path)
        fig, axs = plt.subplots(figsize=(10,10))
        plt.title(f't = {time.strftime("%H:%M")}h')

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
    output_base = Agent_Simulation_V1()
    
    #Sidebar
    st.sidebar.write("## Variables")
    st.sidebar.write("Play with the variables to understand the office utilization and risk 📊🔬")
    what = st.sidebar.selectbox("Analyse", ['Paths', 'Heatmap'], help="Change the Anayses you want to see. Paths: understand where people are walking. Heatmap: See contact hotspots")

    #st.sidebar.write("## Capacity")
    cap = st.sidebar.slider("Number of Employees", min_value=1, max_value=len(output_base), help="Change the capacity and interactively see how the metrics change")

    #st.sidebar.write("## Time")
    time = st.sidebar.slider("Timestep in 15 minutes", min_value=datetime.time(7), max_value=datetime.time(18), step = datetime.timedelta(minutes=15), help="6h-18h, see what is happening in the buildings on a temporal scale")
    t = time.hour*4 + time.minute/15

    st.write
    #site layout
    st.write("""
    # Indoor Simulation Model Omnia

    The overview metric gives insight on what is happening in every variation of the simulation.
    With these scores safety and risk can be assessed in each capacity decision. 
    With the time step slider you can variate throughout the day to see the contact hotspots change on a temporal scale.
    
    Overview Metrics:
    """)

    data = pd.DataFrame.from_dict(json.loads(output_base[str(cap)]))

    metric_row(
            {
                "Capacity": cap,
                "Meters walked": int(sum(data["Activity_timeseries_y"].values[0])/100),
                "Contact per Person": int(data["Contact_score"].values[0]),
            }
        )

    video_slider(t, what, cap, time) 

    explanation = st.beta_expander('What do these metrics mean?', expanded=False)
    explanation.write("""
    The optimization takes the following aspects into account. \n
    1. Contact Score:
    This is the calculated amount of times employees have direct contact in the simulation day. Direct contact means being in the same room and having interaction with a other employee.
    2. Activity Score:
    This refers to the amount of meters walked in the office throughout the day. This is the acticity in the building.
    3. Hygiene Score:
    A metric assessing the infection risk by using direct and indirect contacts between employees as aproxy. indirect contact is employees being in the same space or having touched same surfaces. 
    
    """)

    #Button for more info
    agree = st.checkbox('More')
    if agree:
        fig2, ax = plt.subplots()
        ax.set_title("Variations Simulated")

        #plot time variation
        for i in range(len(data)):
            x = np.array(data["Activity_timeseries_x"].values[i])*4
            y = np.array(data["Activity_timeseries_y"].values[i])/100
            ax.plot(x,y, alpha=0.4) 
        ax.set_xlabel("Time in [15 minutes]")
        ax.set_ylabel("Activity")

        #plot 
        fig3, axs = plt.subplots(1,3, figsize=(10,3))
        fig3.suptitle("Distribution")

        sns.distplot(data["Hygiene_score"], ax=axs[0])
        axs[2].set_yticks([])

        sns.distplot(data["Activity_score"], ax=axs[1])
        axs[0].set_ylabel('')
        axs[0].set_yticks([])

        sns.distplot(data["Contact_score"], ax=axs[2])
        axs[1].set_ylabel('')
        axs[1].set_yticks([])

        #show
        st.pyplot(fig2)
        st.pyplot(fig3)
