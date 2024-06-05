import pandas as pd
import numpy as np
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)
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


time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

print(X_train.shape)
print(y_test.shape)



# need 3d model for stacked lstm
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


from keras.models import Sequential
from keras.layers import LSTM, Dense, Input

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=20,batch_size=64,verbose=1)



test_loss = model.evaluate(X_test, y_test, verbose=0)[0]
# Print the test loss
print("Test loss:", test_loss)




