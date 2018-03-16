from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, SimpleRNN, LSTM

def apply_min_max_scaler(dataset):
  # 終値を 0 ~ 1 の範囲に正規化する
  scaler = MinMaxScaler(feature_range = (0, 1))
  dataset = scaler.fit_transform(dataset)
  return dataset, scaler

def create_dataset(dataset, look_back = 1):
  data_x, data_y = [], []
  for i in range(len(dataset) - look_back):
    data_x.append(dataset[i:(i + look_back), 0])
    data_y.append(dataset[i + look_back, 0])
  np_data_x = np.array(data_x)
  return np.reshape(np_data_x, (np_data_x.shape[0], np_data_x.shape[1], 1)), np.array(data_y)

def create_rnn_model(look_back = 1):
  hid_dim = 10
  model = Sequential()
  model.add(SimpleRNN(hid_dim, input_shape = (look_back, 1)))
  model.add(Dense(1))
  model.compile(loss = 'mean_squared_error', optimizer = 'adam')
  return model

def create_lstm_model(look_back = 1):
  units = 4
  model = Sequential()
  model.add(LSTM(units, input_shape = (look_back, 1)))
  model.add(Dense(1))
  model.compile(loss='mean_squared_error', optimizer='adam')
  return model
