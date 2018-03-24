import sys, os
import time
from sac14.btcusd import bitmex
from sac14.model import apply_min_max_scaler, create_dataset, create_data, create_rnn_model, create_lstm_model
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
  dataframe = bitmex(1)
  dataset = dataframe.values.astype('float32')
  dataset, model_scaler = apply_min_max_scaler(dataset)
  train_x, train_y = create_dataset(dataset, look_back)
  model = create_rnn_model(look_back)
  model.fit(train_x, train_y, epochs=10, batch_size=1, verbose=2)
  model_last_unixtime = dataframe.tail(1).index.values[0]
  model.save(f'{path}/cache/{model_last_unixtime}.h5')
  joblib.dump(model_scaler, f'{path}/cache/{model_last_unixtime}.pkl')

def load():
  global model, model_scaler, model_last_unixtime
  model_last_unixtime = int(sys.argv[2])
  model = load_model(f'{path}/cache/{model_last_unixtime}.h5')
  model_scaler = joblib.load(f'{path}/cache/{model_last_unixtime}.pkl')

def predict():
  while True:
    timer(0)

    while True:
      try:
        dataframe = bitmex(1)
        break
      except:
        print('connection failed')
        time.sleep(15)

    last_close_price = float(dataframe.tail(1)['c'])

    dataset = dataframe.values.astype('float32')
    dataset = model_scaler.fit_transform(dataset)
    data_x = create_data(dataset, look_back)
    prediction = model.predict(data_x)
    prediction = model_scaler.inverse_transform(prediction)
    prediction_close_price = prediction[len(prediction) - 1][0]

    now = datetime.now()
    formatted_now = now.strftime('%Y/%m/%d %H:%M')
    diff = prediction_close_price - last_close_price
    print(f'{formatted_now}, last: {last_close_price}, prediction: {prediction_close_price}, diff: {diff}')

    timer(55)

def timer(x):
  while True:
    if datetime.now().second == x:
      break

def main():
  if sys.argv[1] == 'train':
    train()
    predict()
  elif sys.argv[1] == 'load':
    load()
    predict()

main()