import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import time

def pageZero(sesh):
    #Sidebar
    st.sidebar.write("🧭 After running the simulation navigate to the next page! Navigate through this webapp with the buttons.")
    #Main layout
    st.write("""
    # Omnia Behaviour Package V1.0

    Welcome to Omnia's MVP showcase🎉. In this application you will experience a demo of our innovative simulation model.
    This is the welcome page where we will dive deeper in what you can expect. This application is divided in 3 parts:
    > 1. Run Simulation
    > 2. Examine Result 
    > 3. Optimize Capacity Decision

    Everything is runned real time showing how fast our model has made crowd simulation models.
    We will use a prepackaged [floor plan](https://img.landandplan.com/full/5ad10a8e7efff99a.png) as an example.

    This model uses a behaviour package based on big human behaviour databases from different governmental institutions who have tracked people's behaviour for research purposes.
    Example of one research done on water use can be seen [here](https://www.vewin.nl/publicaties/Paginas/default.aspx).
    Next to these sources we have conducted our own research on building use in our [pilot](https://www.marineterrein.nl/news/omnia-maakt-kantoor-coronaveilig-en-duurzaam/).
    """)

    st.write("Let's start the Simulation🔥:")

    if st.button("Simulate🚀"):
        bar = st.progress(0)
        for i in range(100):
            bar.progress(float(i)/100)
            time.sleep(0.05)
        st.success("Simulation Succeeded -> Go to the next page to check out the results")

    st.write("")
    st.write("")
    st.write("")
    st.write("")

    st.write("""
    Follow us on [Linkedin](https://www.linkedin.com/company/omniamodelling/)

    Check out our [Site](https://www.omniamodelling.com/)

    _All Rights reserved to Omnia_
    """)