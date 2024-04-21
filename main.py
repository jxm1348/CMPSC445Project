import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load the data from CSV file
data = pd.read_csv("electric_vehicle_sales.csv")

# Convert 'Month' column to datetime format, ensuring the format matches your data
data['Date'] = pd.to_datetime(data['Month'], format='%y-%b')

# Converting sales numbers to floats, replacing commas and converting to float
data['BEV'] = data['BEV'].str.replace(',', '').astype(float)
data['PHEV'] = data['PHEV'].str.replace(',', '').astype(float)

# Scaling the features
scaler = MinMaxScaler()
data[['BEV', 'PHEV']] = scaler.fit_transform(data[['BEV', 'PHEV']])

# Function to create sequences
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i-n_steps:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# Number of time steps/number of months we are looking back on to prediction
n_steps = 12

# Preparing data for training the LSTM model
X, y = create_sequences(data[['BEV', 'PHEV']].values, n_steps)

# Splitting data into training and testing sets (80/20 split yeielded the best results imo)
train_size = int(len(X) * 0.80)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Defining the LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(n_steps, 2), return_sequences=True),
    Dropout(0.2),
    LSTM(50, activation='relu'),
    Dropout(0.2),
    Dense(2)  # Predicting both BEV and PHEV sales
])


from tensorflow.keras.optimizers import RMSprop

# Define the model with RMSprop optimizer
model.compile(optimizer=RMSprop(learning_rate=0.001, rho=0.9), loss='mse')


# Early stopping callback to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Training the model with early stopping
model.fit(
    X_train, y_train,
    epochs=100,  # Increased epochs, but early stopping will prevent overfitting
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],  # Include the early stopping callback here
    verbose=1
)

# Prediction
predicted_sales = model.predict(X_test)
predicted_sales = scaler.inverse_transform(predicted_sales)  # Rescaling back to original scale

# might want to also rescale y_test for comparison or further use
y_test_rescaled = scaler.inverse_transform(y_test)

# Showing model structure and summary
model.summary()


import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 'data' still has the 'Date' column correctly set to datetime types
# Extracting the dates corresponding to the test set predictions:
test_dates = data['Date'][-len(predicted_sales):]  # we can adjust index if necessary based on how the data is split

# Figure for BEV Sales Predictions
plt.figure(figsize=(16, 6))  # Increased width for better readability
plt.plot(test_dates, y_test_rescaled[:, 0], label='Actual BEV Sales')
plt.plot(test_dates, predicted_sales[:, 0], label='Predicted BEV Sales')
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.title('BEV Sales Prediction')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.xticks(rotation=45)  # Rotate date labels for better readability
plt.legend()
plt.tight_layout()
plt.show()

# Figure for PHEV Sales Predictions
plt.figure(figsize=(16, 6))  # Separate figure with increased width
plt.plot(test_dates, y_test_rescaled[:, 1], label='Actual PHEV Sales')
plt.plot(test_dates, predicted_sales[:, 1], label='Predicted PHEV Sales')
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.title('PHEV Sales Prediction')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.xticks(rotation=45)  # Rotate date labels for better readability
plt.legend()
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas import date_range

# Function to predict future months
def predict_future_months(model, initial_sequence, n_future_months):
    future_predictions = []
    current_sequence = initial_sequence
    for _ in range(n_future_months):
        next_step = model.predict(current_sequence[np.newaxis, :, :])
        future_predictions.append(next_step[0])
        current_sequence = np.vstack([current_sequence[1:], next_step])
    return np.array(future_predictions)

# Get the last sequence from the data (last n_steps data points)
last_sequence = X[-1]

# Number of future months to predict
n_future_months = 12

# Generate future predictions for BEV and PHEV
future_predictions = predict_future_months(model, last_sequence, n_future_months)

# Rescale predictions back to original scale
future_predictions_rescaled = scaler.inverse_transform(future_predictions)

# Prepare dates for plotting future predictions
last_date = data['Date'].iloc[-1]
future_dates = date_range(start=last_date, periods=n_future_months + 1, freq='M')[1:]

# Plotting the future predictions for BEV in its own window
plt.figure(figsize=(16, 6))
plt.plot(future_dates, future_predictions_rescaled[:, 0], label='Future BEV Sales Prediction', color='blue')
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.title('Future BEV Sales Prediction')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# also plot future PHEV predictions in a separate plot:
plt.figure(figsize=(16, 6))
plt.plot(future_dates, future_predictions_rescaled[:, 1], label='Future PHEV Sales Prediction', color='green')
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.title('Future PHEV Sales Prediction')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
