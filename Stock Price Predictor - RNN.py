# import relevant modules
import pandas as pd
import numpy as np

# read csv file and do some data pre-processing
appl = pd.read_csv('Apple_Price_Model\AAPL.csv')
appl = appl[['Date' , 'Close']]
appl = appl.drop('Date', axis = 1)
appl = appl.reset_index(drop = True)
A = appl.values
A = A.astype('float32')
A = np.reshape(A, (-1,1))

# import sklearn module MinMaxScaler; scaling to optimize convergence speed
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
A = scaler.fit_transform(A)

# split data into training data and testing data
train_size = int(len(A) * 0.80)
test_size = int(len(A) - train_size)
train, test = A[0:train_size,:], A[train_size:len(A),:]

# create features from time series data
def create_features(data, window):
    X, Y = [], []
    for i in range(len(data) - window - 1):
        w = data[i:(i+window), 0]
        X.append(w)
        Y.append(data[i+window, 0])
    return np.array(X), np.array(Y)

# 1mo of trading ~20d (5days/business week)
window_size = 20
X_train, Y_train = create_features(train, window_size)
X_test, Y_test = create_features(test, window_size)

# reshape to [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

A_shape = A.shape
train_shape = train.shape
test_shape = test.shape

def isLeak(A_shape, train_shape, test_shape):
    return not (A_shape[0] == (train_shape[0] + test_shape[0]))

print(isLeak(A_shape, train_shape, test_shape))

# import models
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint

# set seed for reproducibility
tf.random.set_seed(53)
np.random.seed(53)
model = Sequential()
model.add(LSTM(units = 50, activation = 'relu',             # return_sequences = True
    input_shape = (X_train.shape[1], window_size)))
model.add(Dropout(0.2))     # try to prevent overfitting
 
# output layer
model.add(Dense(1))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')

# save models to 'Models' folder
fpath = 'Apple_Price_Model\Models\epoch_{epoch:02d}.hdf5'

chkp = ModelCheckpoint(
    filepath=fpath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

history = model.fit(X_train, Y_train, epochs=150, batch_size = 20, validation_data = (X_test, Y_test), callbacks=[chkp], verbose=1, shuffle=False)

# loading best model (the one chosen below was the best model at the time of this project)
from keras.models import load_model
bis = load_model('Apple_Price_Model\Models\epoch_146.hdf5')

# predicting using test data, inverse transforming said predictions
train_predict = bis.predict(X_train)
yhat_train = scaler.inverse_transform(train_predict)

test_predict = bis.predict(X_test)
yhat_test = scaler.inverse_transform(test_predict)

# inverse xform actual vals, return to original vals
Y_test = scaler.inverse_transform([Y_test])
Y_train = scaler.inverse_transform([Y_train])

# reshape to fit train and test original data
yhat_train = np.reshape(yhat_train, newshape = 985)
yhat_test = np.reshape(yhat_test, newshape = 231)

Y_train = np.reshape(Y_train, newshape=985)
Y_test = np.reshape(Y_test, newshape= 231)

# Model performance evaluation
from sklearn.metrics import mean_squared_error
train_RMSE = np.sqrt(mean_squared_error(Y_train, yhat_train))
test_RMSE = np.sqrt(mean_squared_error(Y_test, yhat_test))

print('Train RMSE is: ' + str(train_RMSE))
print('Test RMSE is: ' + str(test_RMSE))

# visualize models with seaborn and matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# data cleanup to recreate Date column and original close data
aapl = pd.read_csv('Apple_Price_Model\AAPL.csv')
aapl = aapl[['Date', 'Close']]
whole_model = np.hstack((yhat_train, yhat_test))

# viz creation and customization
sns.set_style('darkgrid')
sns.lineplot(data=aapl, x='Date', y='Close', color='b', label='Actual')
sns.lineplot(whole_model, color='r', label='Predicted').set(title='Apple Stock Price vs Predicted', ylabel='Close Price')
plt.legend()
plt.show()
