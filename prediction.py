import pandas as pd
import numpy as np
df=pd.read_csv('/kaggle/input/google-stock-price/GOOG.csv')
df['New_Price'] = (df.splitFactor).replace(np.inf, 1).cumprod() * df.close
df['close'] = df['New_Price'] / 20
df1=df.reset_index()['close']
import matplotlib.pyplot as plt
plt.plot(df1)
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]
