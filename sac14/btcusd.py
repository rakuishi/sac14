import requests
import json
import pandas as pd

def cryptowat():
  res = _get_response()
  labels = ['unixtime', 'open', 'high', 'low', 'close', '_', '_']
  df = pd.DataFrame.from_records(res, columns=labels).set_index('unixtime')
  df.drop(['open', 'high', 'low', '_'], inplace=True, axis=1)
  return _supplement_zero_value(df)

def _get_response():
  periods = 60
  url = f'https://api.cryptowat.ch/markets/bitmex/btcusd-perpetual-futures/ohlc?after=1&periods={periods}'
  res = json.loads(requests.get(url).text)['result'][f'{periods}']
  return res

def _supplement_zero_value(df):
  previous_close = 0
  for index, row in df.iterrows():
    if (row['close'] == 0):
      if (previous_close == 0):
        raise Exception('previous_close is zero.')
      df.at[index, 'close'] = previous_close
    else:
      previous_close = row['close']
  return df
