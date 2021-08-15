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
import time

#imports
def app():
    plt.style.use("bmh")
    image = Image.open('logo_transparant2.png')
    img_path = os.path.abspath('../data/002_test.png')

    #functions
    @st.cache
    def long_running_import():
        # Get output of calculation
        with open("data.json", 'r') as f:
            output_base = json.load(f)
        return output_base

    data = long_running_import()

    @st.cache
    def create_df(data):
        cap = {}
        for i in range(1, len(data)+1):
            df = pd.DataFrame.from_dict(json.loads(data[str(i)]))
            
            HS = df.Hygiene_score.mean()
            CONT = df.Contact_score.mean()
            ACT = df.Activity_score.mean()

            cap[i] = [HS, CONT,ACT]

        newdf = pd.DataFrame.from_dict(cap).T
        newdf.columns = ['Hygiene Score','Direct Contact','Activity']
        newdf['Capacity'] = newdf.index
        return newdf
    df = create_df(data)

    def optimize(df, area=250, alpha1=0.4, alpha2=0.6, alpha3=0.8):
        def normalize(series):
            max_val = np.max(series)
            return series / max_val

        HS = 1-normalize(df['Hygiene Score'])
        DC = 1-normalize(df['Direct Contact'])
        ACT = 1-normalize(df['Activity'])

        advice_cap = area * 0.05
        cap_score = list(np.abs(df.index - advice_cap)*-1+1)

        #cap_score = cap_score + np.abs(np.min(cap_score))
        
        total_score = list(HS* alpha1 + DC*alpha2 + ACT*alpha3 + cap_score)

        sort_index = np.argsort(total_score)
        best = np.argmax(total_score)

        return sort_index, best, total_score, [HS[best], DC[best], ACT[best], cap_score[best]]

    #opzet van pagina
    st.write("""# Optimization of Capacity

    This is a decision support tool helping you to find the optimal capacity level of your office building. Based on a crowd simulation model, the optimal capacity is decided on three aspects. \n 
    1. Hygiene score: This is related to the amount of indirect contacts between employees.
    2. Direct Contact: The amount of direct contact between employees. This heavily impacts the infection safety.
    3. The total amount of acitivty.
    4. The number of square meters. This has the biggest influence on capacity.

    On the left of the screen you can see a number of questions. This helps us in understanding which element of the capacity calculation you find most valauble. This changes the way the calculations are done and which advice is provided.
    """)

    area = st.sidebar.text_input("Area in square meters", max_chars=6, value=100, help='Number of Square Feet Divided by 10.8')
    safety = st.sidebar.slider("How important is employee safety?", min_value=1, max_value=10, value=5, help='This includes Covid infection Safety')
    prod = st.sidebar.slider("How is important is employee productivity?", min_value=1, max_value=10, value=5, help="When the capacity is too high employees' productivity tends to decline")

    #make copy to not overwrite main file
    table_data = df.copy() 
    test = st.dataframe(table_data)


    #button for erasing non optimal capacities
    if st.sidebar.button("optimize"):
        indlist, best, score, KPIS = optimize(table_data, int(area), safety/10, safety/10, prod/10 )
        indlist = np.delete(indlist, -1)
        
        indlist_fresh = []

        bar = st.progress(0)
        for i in range(len(indlist)):
            indlist_fresh.append(indlist[i]+1)
            test.write(table_data.drop(indlist_fresh))
            bar.progress(float(i+2)/21)
            time.sleep(0.1)
        st.success("Optimization Succeeded")

        st.write("### Optimal capacity")
        metric("Capacity", best+1)
        st.write("### Most Valuable Metrics")
        metric_row(
            {
                'Infection Risk': round(0.5*(KPIS[0]+KPIS[1]),3),
                'Productivity': round(KPIS[2], 3),
                'Cap_score': round(KPIS[3],3)
            }
        )

        explanation = st.beta_expander('What do these metrics mean?', expanded=False)
        explanation.write("""
        The optimization takes the following aspects into account. \n
        1. Infection risk:
        This is related to the number of direct and indirect contact between employees.
        2. Productivity:
        This refers to time wasted due the high numer of people.
        3. Cap_score:
        An estimate of the capacity can be given by use of a rule of thumb. Capacity = m2 * 0.05
        
        """)

        #maak dataframe with scores
        plot_data = {'capacity': np.arange(1, len(score)+1), 'Score': score}
        plot_data = pd.DataFrame(plot_data)
        
        #plot scores
        fig2, ax = plt.subplots()
        plot_data.plot(x="capacity",y="Score", ax=ax)
        table_data.drop("Capacity", axis=1).plot(ax=ax, secondary_y=["Direct Contact","Activity"], legend=True)
        st.pyplot(fig=fig2)

    #more inforamtion for original metrics
    if st.checkbox("More"):
        fig2, ax = plt.subplots(1,2)
        sns.distplot(table_data["Hygiene Score"], ax=ax[0])
        sns.distplot(table_data["Activity"], ax=ax[1])
        st.pyplot(fig=fig2)
 
