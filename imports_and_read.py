#Import the libraries
import os
import math
from dateutil.relativedelta import relativedelta
#import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from tensorflow.keras.models import load_model
from nsepy import get_history
from datetime import date
import datetime
import streamlit as st

from PIL import Image
