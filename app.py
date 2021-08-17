import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.text import Annotation
import numpy as np
import pandas as pd
import os
import json
import time
from PIL import Image
import base64

from apps import Home, Results, Optimize, SessionState, heatmap

import streamlit as st
import io

st.set_page_config(page_title="Omnia MVP", page_icon=":rocket:")

sesh = SessionState.get(curr_page = 0)
PAGES = [Home.pageZero, Results.pageOne, Optimize.pageTwo]

def main():
    ####SIDEBAR STUFF
    image = Image.open('data/logo_transparant2.png')
    link = "https://www.omniamodelling.com/"

    @st.cache(allow_output_mutation=True)
    def get_base64_of_bin_file(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()

    @st.cache(allow_output_mutation=True)
    def get_img_with_href(local_img_path, target_url):
        img_format = os.path.splitext(local_img_path)[-1].replace('.', '')
        bin_str = get_base64_of_bin_file(local_img_path)
        html_code = f'''
            <a href="{target_url}" target="_blank">
                <img src="data:image/{img_format};base64,{bin_str}" width=200px height=auto/>
            </a>'''
        return html_code

    gif_html = get_img_with_href("data/logo_transparant2.png", link)
    st.sidebar.markdown(gif_html, unsafe_allow_html=True)

    #####MAIN PAGE NAV BAR:
    st.sidebar.markdown(' # Navigation')
    st.sidebar.markdown(' ')
    if sesh.curr_page > 0:
        if st.sidebar.button('Back'):
            sesh.curr_page = max(0, sesh.curr_page-1)
    if st.sidebar.button('Next page'):
        sesh.curr_page = min(len(PAGES)-1, sesh.curr_page+1)

    #####MAIN PAGE APP:
    page_turning_function = PAGES[sesh.curr_page]
    page_turning_function(sesh)

if __name__=='__main__':
    main()
