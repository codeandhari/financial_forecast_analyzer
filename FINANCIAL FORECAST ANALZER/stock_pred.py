import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Setting plot parameters
rcParams['figure.figsize'] = 20, 10

# Read data
df = pd.read_csv("NSE-TATA.csv")

# Data preprocessing
df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
df.index = df['Date']
plt.figure(figsize=(16, 8))
plt.plot(df["Close"], label='Close Price history')

# Creating new dataset
new_dataset = df.sort_index(ascending=True, axis=0)
new_dataset = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])

for i in range(0, len(new_dataset)):
    new_dataset["Date"][i] = new_dataset.index[i]
    new_dataset["Close"][i] = df["Close"][i]

new_dataset.index = new_dataset.Date
new_dataset.drop("Date", axis=1, inplace=True)

# Scaling data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(new_dataset.values)

# Splitting data into train and validation sets
train_data = scaled_data[:987, :]
valid_data = scaled_data[987:, :]

# Creating train data
x_train, y_train = [], []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Building LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

# Prepare test data and predict
inputs = new_dataset[len(new_dataset) - len(valid_data) - 60:].values
inputs = scaler.transform(inputs)

X_test = []
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i-60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

# Save model
model.save("saved_lstm_model.h5")

# Plotting results
train = new_dataset[:987]
valid = new_dataset[987:]
valid['Predictions'] = closing_price
plt.plot(train["Close"])
plt.plot(valid[['Close', "Predictions"]])
plt.show()
