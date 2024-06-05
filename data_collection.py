import pandas_datareader as pdr
import pandas as pd
df=pdr.get_data_tiingo('GOOG',api_key='8eef2d5c370bc76d47b80d774e6f0f6ee507a0a9')
df.to_csv('GOOG.csv')
df=pd.read_csv('GOOG.csv')
print(df.head())