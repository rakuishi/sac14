import sys, os
from sac14.btcusd import bitmex
from sac14.model import apply_min_max_scaler, create_dataset, create_rnn_model, create_lstm_model
from datetime import datetime
from sklearn.externals import joblib
from keras.models import load_model

path = os.path.dirname(os.path.abspath(__file__))
look_back = 3

model = None
model_scaler = None
model_last_unixtime = 0

def train():
  global model, model_scaler, model_last_unixtime
  dataframe = bitmex()
  dataset = dataframe.values.astype('float32')
  dataset, model_scaler = apply_min_max_scaler(dataset)
  train_x, train_y = create_dataset(dataset, look_back)
  model = create_rnn_model(look_back)
  model.fit(train_x, train_y, epochs=5, batch_size=1, verbose=2)
  model_last_unixtime = dataframe.tail(1).index.values[0]
  model.save(f'{path}/cache/{model_last_unixtime}.h5')
  joblib.dump(model_scaler, f'{path}/cache/{model_last_unixtime}.pkl')

def load():
  global model, model_scaler, model_last_unixtime
  model_last_unixtime = int(sys.argv[2])
  model = load_model(f'{path}/cache/{model_last_unixtime}.h5')
  model_scaler = joblib.load(f'{path}/cache/{model_last_unixtime}.pkl')

def predict():
  dataframe = bitmex()
  print(dataframe.tail(1))

def main():
  if sys.argv[1] == 'train':
    train()
    predict()
  elif sys.argv[1] == 'load':
    load()
    predict()

main()
