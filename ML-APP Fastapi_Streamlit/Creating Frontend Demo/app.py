#!/usr/bin/env python
# coding: utf-8

import os
import streamlit as st
import numpy as np
import pandas as pd
import pycaret
from pycaret.classification import *
import joblib
from joblib import load
import streamlit as st
import _pickle as pickle
from pprint import pformat
from pycaret.regression import *
from PIL import Image
import markdown
import requests



def main():

        """SAI ML App """
        image = Image.open('sai_app_header.png')
        st.image(image,use_column_width=True)
        with open("README.md", "r", encoding="utf-8") as input_file:
           text = input_file.read()
        intro_markdown=markdown.markdown(text)
        st.markdown(intro_markdown, unsafe_allow_html=True)

        image_1 = Image.open('data_intro.png')
        st.image(image_1,use_column_width=True)

        st.subheader("Click On Predict to know the Diamond Price (USD)")

        import pandas as pd
        df=pd.read_csv("diamond.csv")
        temp = df.to_dict('list')
        temp['Cut'] = list(set(temp['Cut']))
        temp['Color'] = list(set(temp['Color']))
        temp['Clarity'] = list(set(temp['Clarity']))
        temp['Polish'] = list(set(temp['Polish']))
        temp['Symmetry'] = list(set(temp['Symmetry']))
        temp['Report'] = list(set(temp['Report']))
        temp_records = df.to_dict('records')

        st.sidebar.markdown("## Select Metrics Below")
        #CARAT WEIGHT
        Carat_Weight_list = st.sidebar.number_input('Carat Weight')
        #Checkbox for CUT
        CUT_list = st.sidebar.selectbox("Select CUT", temp['Cut'])
        #Checkbox for Color
        Color_list = st.sidebar.selectbox("Select Color", temp['Color'])
        #Checkbox for Clarity
        Clarity_list = st.sidebar.selectbox("Select Clarity", temp['Clarity'])
        #Checkbox for Polish
        Polish_list = st.sidebar.selectbox("Select Polish", temp['Polish'])
        #Checkbox for Symmetry
        Symmetry_list = st.sidebar.selectbox("Select Symmetry", temp['Symmetry'])
        #Checkbox for Report
        Report_list = st.sidebar.selectbox("Select Report", temp['Report'])


        cols = ['Carat Weight', 'Cut', 'Color','Clarity','Polish','Symmetry','Report']
        # store the inputs
        features = [Carat_Weight_list, CUT_list, Color_list,Clarity_list,Polish_list,Symmetry_list,Report_list]
        # fastapi endpoint
        url = 'http://127.0.0.1:8000'
        endpoint = '/api_diamond/'

        def process(Carat_Weight_list, CUT_list,Color_list,Clarity_list,Polish_list,Symmetry_list,Report_list, server_url: str):
            r = requests.get(server_url, params={"Carat_Weight_list": Carat_Weight_list, "CUT_list": CUT_list,"Color_list": Color_list,"Clarity_list": Clarity_list,"Polish_list": Polish_list,"Symmetry_list": Symmetry_list,"Report_list": Report_list})
            return r


        if st.button('Predict'):
            result = process(Carat_Weight_list, CUT_list,Color_list,Clarity_list,Polish_list,Symmetry_list,Report_list, url + endpoint)
            if (result.status_code == 200):
                jsonDecoded = result.json()
                pred=jsonDecoded["prediction"]
                st.balloons()
                st.success(f'Price is: {pred} ')







if __name__ == '__main__':
	main()
