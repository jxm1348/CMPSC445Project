import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas import date_range
from sklearn.metrics import mean_squared_error

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

# Add a custom callback to print loss after each epoch
class LossHistory(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        print("Epoch:", epoch + 1, "Loss:", logs['loss'], "Validation Loss:", logs['val_loss'])


# Training the model with early stopping
model.fit(
    X_train, y_train,
    epochs=100,  # Increased epochs, but early stopping will prevent overfitting
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, LossHistory()],  # Include the early stopping callback here
    verbose=1
)

# Prediction
predicted_sales = model.predict(X_test)
predicted_sales = scaler.inverse_transform(predicted_sales)  # Rescaling back to original scale

# might want to also rescale y_test for comparison or further use
y_test_rescaled = scaler.inverse_transform(y_test)

print("Actual Sales:", y_test_rescaled)
print("Predicted Sales:", predicted_sales)

# Calculate and print MSE
mse = mean_squared_error(y_test_rescaled, predicted_sales)
print("Mean Squared Error on Test Set:", mse)

# Showing model structure and summary
model.summary()

# Print model summary
model.summary(print_fn=lambda x: print(x))



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

# Print predictions
#print("Actual Sales:", y_test_rescaled)
#print("Predicted Sales:", predicted_sales)

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

# Print predictions
#print("Actual Sales:", y_test_rescaled)
#print("Predicted Sales:", predicted_sales)


# Function to predict future months
def predict_future_months(model, initial_sequence, n_future_months):
    future_predictions = []
    current_sequence = initial_sequence
    for _ in range(n_future_months):
        if isinstance(model, Sequential):  #if LSTM model
            next_step = model.predict(current_sequence[np.newaxis, :, :])
        else:  # If linear regression
            next_step = model.predict(current_sequence.reshape(1, -1))
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

# Print future predictions to console
print("Future Predictions for BEV and PHEV:")
for date, prediction in zip(future_dates, future_predictions_rescaled):
    print(f"{date.strftime('%Y-%m')}: BEV={prediction[0]}, PHEV={prediction[1]}")

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

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)


# Training linear regression model
linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train_flat, y_train)

# Make predictions using linear regression
predicted_sales_linear = linear_reg_model.predict(X_test_flat)

# Rescale predictions back to original scale
predicted_sales_linear_rescaled = scaler.inverse_transform(predicted_sales_linear)

# Compare the predictions of linear regression with LSTM model
print("Mean Squared Error on Test Set (Linear Regression):", mean_squared_error(y_test_rescaled, predicted_sales_linear_rescaled))

# Plotting the comparison of LSTM and linear regression predictions for BEV
plt.figure(figsize=(16, 6))
plt.plot(test_dates, y_test_rescaled[:, 0], label='Actual BEV Sales', color='blue')
plt.plot(test_dates, predicted_sales[:, 0], label='LSTM Predicted BEV Sales', linestyle='--', color='red')
plt.plot(test_dates, predicted_sales_linear_rescaled[:, 0], label='Linear Regression Predicted BEV Sales', linestyle='--', color='green')
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.title('BEV Sales Prediction Comparison (LSTM vs Linear Regression)')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Plotting the comparison of LSTM and linear regression predictions for PHEV
plt.figure(figsize=(16, 6))
plt.plot(test_dates, y_test_rescaled[:, 1], label='Actual PHEV Sales', color='blue')
plt.plot(test_dates, predicted_sales[:, 1], label='LSTM Predicted PHEV Sales', linestyle='--', color='red')
plt.plot(test_dates, predicted_sales_linear_rescaled[:, 1], label='Linear Regression Predicted PHEV Sales', linestyle='--', color='green')
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.title('PHEV Sales Prediction Comparison (LSTM vs Linear Regression)')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Number of future months to predict
n_future_months = 12

# Get the last sequence from the data (last n_steps data points)
last_sequence_linear = X[-1]

# Generate future predictions for BEV and PHEV using linear regression model
future_predictions_linear = predict_future_months(linear_reg_model, last_sequence_linear, n_future_months)

# Rescale predictions back to original scale
future_predictions_linear_rescaled = scaler.inverse_transform(future_predictions_linear)

# Prepare dates for plotting future predictions
last_date = data['Date'].iloc[-1]
future_dates_linear = date_range(start=last_date, periods=n_future_months + 1, freq='M')[1:]

# Print future predictions to console
print("Future Predictions for BEV and PHEV (Linear Regression):")
for date, prediction in zip(future_dates_linear, future_predictions_linear_rescaled):
    print(f"{date.strftime('%Y-%m')}: BEV={prediction[0]}, PHEV={prediction[1]}")

# Reshape future predictions from LSTM model for compatibility with linear regression predictions
future_predictions_lstm_rescaled = future_predictions_rescaled.reshape(-1, 2)

# Plotting the comparison of LSTM and linear regression predictions for BEV
plt.figure(figsize=(16, 6))
plt.plot(future_dates_linear, future_predictions_linear_rescaled[:, 0], label='Linear Regression BEV Sales Prediction', linestyle='--', color='red')
plt.plot(future_dates, future_predictions_lstm_rescaled[:, 0], label='LSTM BEV Sales Prediction',  color='blue')
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.title('BEV Sales Prediction Comparison (Linear Regression vs LSTM)')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Plotting the comparison of LSTM and linear regression predictions for PHEV
plt.figure(figsize=(16, 6))
plt.plot(future_dates_linear, future_predictions_linear_rescaled[:, 1], label='Linear Regression PHEV Sales Prediction', linestyle='--',color='yellow')
plt.plot(future_dates, future_predictions_lstm_rescaled[:, 1], label='LSTM PHEV Sales Prediction',  color='green')
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.title('PHEV Sales Prediction Comparison (Linear Regression vs LSTM)')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()