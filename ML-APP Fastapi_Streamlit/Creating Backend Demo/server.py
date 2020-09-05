
#Creating backend for streamlit Application
import os
import streamlit as st
import numpy as np
import pandas as pd
import pycaret
from pycaret.regression import *
import joblib
from joblib import load
import streamlit as st
import _pickle as pickle
from pprint import pformat
from pycaret.regression import *
from PIL import Image
import markdown
from typing import Optional
from fastapi import FastAPI
from ratelimit import limits

app = FastAPI()

# Use pickle to load in the pre-trained model
model = pycaret.regression.load_model('Sai_LGBM')

@app.get("/")
def read_root():
	return {"Diamond Rate Prediction Application": "FAST API"}

SECONDS = 60
@app.get("/api_diamond/")
@limits(calls=5, period=SECONDS)
def predict(Carat_Weight_list:float,CUT_list:str,Color_list:str,Clarity_list:str,Polish_list:str,Symmetry_list:str,Report_list:str):
    #print(payload)
    #features
    cols = ['Carat Weight', 'Cut', 'Color','Clarity','Polish','Symmetry','Report']
    # store the inputs
    features = [Carat_Weight_list, CUT_list, Color_list,Clarity_list,Polish_list,Symmetry_list,Report_list]
    data_unseen=pd.DataFrame([features],columns=cols)
    prediction=predict_model(model,data=data_unseen)
    pred=(prediction.Label[0])
    ret = {"prediction":int(pred)}
    return ret
