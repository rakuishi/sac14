import sys, os, time, ccxt
from datetime import datetime
from os.path import join, dirname
from sac14.btcusd import bitmex
from sac14.model import apply_min_max_scaler, create_dataset, create_data, create_rnn_model, create_lstm_model
from sklearn.externals import joblib
from keras.models import load_model
from keras.callbacks import EarlyStopping
from dotenv import load_dotenv

path = os.path.dirname(os.path.abspath(__file__))
look_back = 3
debug = True
print(f'Debug: {debug}')

dotenv_path = path + '/.env'
load_dotenv(dotenv_path)
ccxt = ccxt.bitmex({
  'apiKey': os.environ.get('TESTNET_API_KEY'),
  'secret': os.environ.get('TESTNET_SECRET'),
})
if debug:
  ccxt.urls['api'] = ccxt.urls['test']
print(f"Using: {ccxt.urls['api']}")

model = None
model_scaler = None
model_last_unixtime = 0

def train(dataframe = None):
  global model, model_scaler, model_last_unixtime
  if dataframe is None:
    dataframe = bitmex(5, debug)
  dataset = dataframe.values.astype('float32')
  dataset, model_scaler = apply_min_max_scaler(dataset)
  train_x, train_y = create_dataset(dataset, look_back)
  model = create_rnn_model(look_back)
  early_stopping = EarlyStopping(monitor='val_loss', patience=2)
  model.fit(train_x, train_y, epochs=5, batch_size=1, verbose=2, callbacks=[early_stopping])

def predict():
  while True:
    timer(0)

    connection_success = True
    try:
      dataframe = bitmex(5, debug)
      connection_success = True
    except:
      print('connection failed')
      connection_success = False
      time.sleep(1)

    if connection_success:
      train(dataframe)
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

def timer(x):
  while True:
    now = datetime.now()
    if now.second == x and now.minute % 5 == 0:
      break
    time.sleep(0.8)

def main():
  if sys.argv[1] == 'predict':
    predict()

main()