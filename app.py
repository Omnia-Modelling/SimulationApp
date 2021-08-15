import streamlit as st
from multiapp import MultiApp
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.text import Annotation
import numpy as np
import pandas as pd
import seaborn as sns
from streamlit_metrics import metric, metric_row 
import os
import json
from PIL import Image

from apps import Simulate, Capacity, main

import warnings
warnings.filterwarnings("ignore")


app = MultiApp()

st.set_page_config(page_title="Omnia Simulation Model", page_icon="🚀")

image = Image.open('logo_transparant2.png')
img_path = os.path.abspath('../data/002_test.png')


app.add_app('Home', main.app)
app.add_app("Simulate", Simulate.app)
app.add_app("Capacity", Capacity.app)

app.run()
