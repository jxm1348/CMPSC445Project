import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

# Loading in data from csv
data = pd.read_csv("electric_vehicle_sales.csv")

# Preprocessing
data['Date'] = pd.to_datetime(data['Month'], format='%y-%b')

# Separating year and month from yy-month
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year

# Converting sales to numerical vals and getting rid of commas (1,700 -> 1700)
data['BEV'] = data['BEV'].str.replace(',', '').astype(float)
data['PHEV'] = data['PHEV'].str.replace(',', '').astype(float)

# Feature scaling
scaler = MinMaxScaler()
data[['BEV_scaled', 'PHEV_scaled']] = scaler.fit_transform(data[['BEV', 'PHEV']])

print(data)

