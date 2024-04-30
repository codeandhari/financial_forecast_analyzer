import pandas as pd
import numpy as np
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Load the dataset
df = pd.read_csv('NSE-TATA.csv')

# Preprocess the data
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.dropna(inplace=True)

# Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1,1))

# Split the data into training and validation sets
training_size = int(len(scaled_data) * 0.8)
test_size = len(scaled_data) - training_size
train_data, test_data = scaled_data[0:training_size,:], scaled_data[training_size:len(scaled_data),:]

# Reshape the data
def create_dataset(dataset, time_steps=1):
    X, Y = [], []
    for i in range(len(dataset)-time_steps-1):
        a = dataset[i:(i+time_steps), 0]
        X.append(a)
        Y.append(dataset[i + time_steps, 0])
    return np.array(X), np.array(Y)
    
time_steps = 100
X_train, Y_train = create_dataset(train_data, time_steps)
X_test, Y_test = create_dataset(test_data, time_steps)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(time_steps, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='binary_crossentropy')

# Train the model
model.fit(X_train, Y_train, epochs=1, batch_size=1, verbose=2)

# Evaluate the model
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
Y_train = scaler.inverse_transform([Y_train])
test_predict = scaler.inverse_transform(test_predict)
Y_test = scaler.inverse_transform([Y_test])

train_score = np.sqrt(mean_squared_error(Y_train[0], train_predict[:,0]))
test_score = np.sqrt(mean_squared_error(Y_test[0], test_predict[:,0]))

# Create the Dash app
app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='Stock Price Prediction'),
    dcc.Graph(
        id='train-test-predict',
        figure={
            'data': [
                {'x': df.index[0:len(train_data)], 'y': Y_train[0], 'type': 'line', 'name': 'Train Data'},
                {'x': df.index[len(train_data)+2*time_steps:], 'y': Y_test[0], 'type': 'line', 'name': 'Test Data'},
                {'x': df.index[time_steps:len(train_data)], 'y': train_predict[:,0], 'type': 'line', 'name': 'Train Predictions'},
                {'x': df.index[len(train_data)+2*time_steps:len(scaled_data)-1], 'y': test_predict[:,0], 'type': 'line', 'name': 'Test Predictions'}
            ],
            'layout': {
                'title': 'Train and Test Data'
            }
        }
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)