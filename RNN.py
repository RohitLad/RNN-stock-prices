import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
# Import data
train_data = pd.read_csv('Google_Stock_Price_Train.csv')
train_data = train_data.iloc[:,1:2].values # opening prices

# Scaling
sc = MinMaxScaler()
train_data = sc.fit_transform(train_data)

# Manage i/p and o/p
X_train = train_data[0:1257]
y_train = train_data[1:1258]

# reshape because of expected 3D tensor

X_train = np.reshape(X_train, (1257, 1, 1))

# Initialization
reg_model = Sequential()
reg_model.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))
reg_model.add(Dense(units=1))
reg_model.compile(optimizer = 'adam', loss = 'mean_squared_error')
reg_model.fit(X_train,y_train,batch_size = 32, epochs = 200)

# Prediction
test_data = pd.read_csv('Google_Stock_Price_Test.csv')
test_data = test_data.iloc[:,1:2].values

# Get predicted stock price
ip = test_data
ip = sc.transform(ip)
ip = np.reshape(ip,(20,1,1))
pred = reg_model.predict(ip)
pred = sc.inverse_transform(pred)

# Plot
plt.plot(test_data, color = 'red', label = 'Real price')
plt.plot(pred, color = 'blue', label = 'Predicted price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
