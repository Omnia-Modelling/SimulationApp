# Python packages
import matplotlib.pyplot as plt
from matplotlib.text import Annotation
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from streamlit_metrics import metric, metric_row 
from zipfile import ZipFile
import os
import json
from PIL import Image
import time
import scipy



def pageTwo(sesh):
    plt.style.use("bmh")
    image = Image.open('data/logo_transparant2.png')
    img_path = os.path.abspath('../data/002_test.png')

    #functions
    @st.cache
    def long_running_import():
        d = None  
        data = None  
        with ZipFile("data/data.zip", "r") as z:
            for filename in z.namelist():   
                with z.open(filename) as f:  
                    data = f.read()  
                    d = json.loads(data.decode("utf-8")) 
        return d

    @st.cache
    def Launching_AI(data):
        cap = {}
        for i in range(1, len(data)+1):
            df = pd.DataFrame.from_dict(json.loads(data[str(i)]))
            
            HS = df.Hygiene_score.mean()
            CONT = df.Contact_score.mean()
            ACT = df.Activity_score.mean()

            cap[i] = [HS, CONT,ACT]

        newdf = pd.DataFrame.from_dict(cap).T
        newdf.columns = ['Infection Risk','Direct Contact','Activity Score']
        newdf['Capacity'] = newdf.index
        return newdf

    def optimize(df, risk):
        def normalize(series, a,b):
            max_val = np.max(series)
            min_val = np.min(series)
            return ((b-a)*(series-min_val) / (max_val-min_val))+a
        def schaal(series, a,b):
            max_val = 1
            min_val = 0
            return ((b-a)*(series-min_val) / (max_val-min_val))+a

        HS = 1-normalize(df['Infection Risk'].values, a=0, b=1)
        DC = 1-normalize(df['Direct Contact'].values, a=0, b=1)
        ACT = 1-normalize(df['Activity Score'].values, a=0, b=1)
        Safety = (HS*0.3+DC*0.6+ACT*0.1)

        area = 470
        Advice_cap = area * 0.04
        
        cap_dec = [number - Advice_cap for number in list(range(1, len(Safety)+1))]
        cap_dec = np.abs(cap_dec)*-1
        cap_dec = normalize(cap_dec, a=0, b=1)

        def maprisk(inp):
            if inp == 'Productivity':
                return 1
            if inp == "Safety":
                return 10
            else:
                return inp

        alpha = (schaal(maprisk(risk)/10, a=0.25, b=0.75)  +0.5)/2
        
        safetyscore = list(Safety*alpha)
        safetyscore = safetyscore
        cap_dec = cap_dec*(1-alpha)
        total_score = safetyscore + cap_dec

        sort_index = np.argsort(total_score)
        best = np.argmax(total_score)

        return sort_index, best, total_score, [HS[best], DC[best], ACT[best]], cap_dec, safetyscore

    #data creation.
    data = long_running_import()
    table_data = Launching_AI(data)

    #opzet van pagina
    st.write("""
    # Optimization of Capacity

    This is a decision support tool helping you find the optimal capacity level of your office building. Based on a crowd simulation model, the optimal capacity is decided on three aspects. \n 
    1. Infection Risk: This is related to the amount of indirect contacts between employees.
    2. Direct Contact: The amount of direct contact between employees. This heavily impacts the infection safety.
    3. Activity Score: The total amount of activity in the office based on how much employees are walking around.
    4. The number of square meters of the building in square meters [m^2]. This has the biggest influence on capacity.

    On the left of the screen you can see a number of questions. This helps us in understanding which element of the capacity calculation you find most valuable. This changes the way the calculations are done and which advice is provided.
    """)

    st.sidebar.write("# Input")
    st.sidebar.write("Tell us about your building and what your priorities are. Optimize your capacity decision 📈")

    opt = ["Productivity", "Safety"]
    range_l = list(range(1,10+1))
    range_l[0], range_l[-1]= opt[0], opt[-1]

    risk = st.sidebar.select_slider('Select how risk tolerant you are', range_l, help="Choose a capacity strategy")
    test = st.dataframe(table_data)

    #button for erasing non optimal capacities
    if st.sidebar.button("Optimize"):
        indlist, best, score, KPIS,caps,saf = optimize(table_data, risk)
        indlist = np.delete(indlist, -1)
        
        indlist_fresh = []

        bar = st.progress(0)
        for i in range(len(indlist)):
            indlist_fresh.append(indlist[i]+1)
            test.write(table_data.drop(indlist_fresh))
            bar.progress(float(i+2)/21)
            time.sleep(0.1)
        a = st.success("Optimization Succeeded")

        st.write("### Optimal capacity")
        metric("Capacity", best+1)
        st.write("### Most Valuable Metrics")

        metric_row(
            {
                'Infection Risk': round((KPIS[0]+2*KPIS[1])/3,3),
                'Productivity': round(KPIS[2], 3),
                'CAP_Rule of Thumb': int(470*0.03)
            }
        )

        explanation = st.beta_expander('What do these metrics mean?', expanded=False)
        explanation.write("""
        The optimization takes the following aspects into account. \n
        1. Infection risk:
        This is related to the number of direct and indirect contact between employees.
        2. Productivity:
        This refers to time wasted due to the high number of people.
        3. Cap_score:
        An estimate of the capacity can be given by use of a rule of thumb. Capacity = Area * 0.05
        
        """)

        #maak dataframe with scores
        plot_data = {'capacity': np.arange(1, len(score)+1), 'Total Score': score, 'Rule of Thumb':caps, 'Infection Safety':saf}
        plot_data = pd.DataFrame(plot_data)
        
        #plot scores
        fig2, ax = plt.subplots(2,1)
        ax[0].vlines(best+1, ymin=table_data["Infection Risk"].min()-10, ymax=table_data["Infection Risk"].max()+10, linestyle="--", color='red')
        table_data.drop("Capacity", axis=1).plot(ax=ax[0], secondary_y=["Direct Contact","Activity Score"], legend=True)
        ax[0].set_ylim(table_data["Infection Risk"].min()-10, table_data["Infection Risk"].max()+10)

        plot_data.plot(x="capacity",y="Total Score", ax=ax[1], lw=2, color='red', alpha=0.5)
        plot_data.plot(x="capacity",y="Rule of Thumb", ax=ax[1], lw=2, color='blue', alpha=0.5)
        plot_data.plot(x="capacity",y="Infection Safety", ax=ax[1], lw=2, color='yellow', alpha=0.5)
        
        ax[0].set_title("Optimal")
        st.pyplot(fig=fig2)

    #nodig anders is checkbox pre checked
    agree = st.checkbox("More")
    #more inforamtion for original metrics
    if agree:
        fig2, ax = plt.subplots(1,3, figsize=(10,3))
        sns.distplot(table_data["Infection Risk"], ax=ax[0])
        sns.distplot(table_data["Activity Score"], ax=ax[1])
        sns.distplot(table_data["Direct Contact"], ax=ax[2])
        ax[1].set_ylabel("")
        ax[2].set_ylabel("")
        ax[1].set_yticks([])
        ax[2].set_yticks([])
        ax[0].set_yticks([])
        st.pyplot(fig=fig2)

    st.write("_all rights reserved to Omnia V.O.F._")