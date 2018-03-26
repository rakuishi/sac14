import requests
import json
import pandas as pd
from datetime import datetime, timedelta

def cryptowat(periods = 300):
  url = f'https://api.cryptowat.ch/markets/bitmex/btcusd-perpetual-futures/ohlc?after=1&periods={periods}'
  res = json.loads(requests.get(url).text)['result'][f'{periods}']
  labels = ['unixtime', 'open', 'high', 'low', 'close', '_', '_']
  df = pd.DataFrame.from_records(res, columns=labels).set_index('unixtime')
  df.drop(['open', 'high', 'low', '_'], inplace=True, axis=1)
  return _supplement_zero_value(df)

def bitmex(resolution = 5):
  now = datetime.now()
  to_unixtime = now.strftime('%s')
  from_unixtime = (now - timedelta(weeks=1)).strftime('%s')
  url = f'https://www.bitmex.com/api/udf/history?symbol=XBTUSD&resolution={resolution}&from={from_unixtime}&to={to_unixtime}'
  res = json.loads(requests.get(url).text)
  df = pd.DataFrame(data=res).set_index('t')
  df.drop(['h', 'l', 'o', 's', 'v'], inplace=True, axis=1)
  return df

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
