from fastapi import FastAPI
import joblib 
import numpy as np


model=joblib.load('model.joblib')