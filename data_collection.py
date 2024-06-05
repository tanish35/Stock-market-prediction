import pandas_datareader as pdr
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()
api_key = os.getenv('TIINGO_API_KEY')
df=pdr.get_data_tiingo('GOOG',api_key=api_key)
df.to_csv('GOOG.csv')
df=pd.read_csv('GOOG.csv')
print(df.head())