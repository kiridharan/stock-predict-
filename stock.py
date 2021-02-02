import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt


from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


# load Data

company = 'AAPL'

start = dt.datetime(2018,1,1)
end = dt.datetime(2020,2,2)

data =web.DataReader(company , 'yahoo' , start , end)

# prepare Data

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

prediction_days = 60

x_train = []
y_train = []

for x in range(prediction_days , len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x])
    y_train.append(scaled_data[x,0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] ,1))

# build the model

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=70, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=70))
model.add(Dropout(0.2))
model.add(Dense(units=1))


model.compile(optimizer='adam' , loss='mean_squared_error')
model.fit(x_train , y_train , epochs=30 , batch_size=40)


''' TEst the model accuraacy on exting Data'''

test_start = dt.datetime(2018,1,1)
test_end = dt.datetime.now()

test_data = web.DataReader(company , 'yahoo' , test_start , test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']) , axis=0)

model_input = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_input = model_input.reshape(-1,1)

model_input = scaler.transform(model_input)

# make predictionon test data

x_test = []

for x in range(prediction_days , len(model_input)):
    x_test.append(model_input[x-prediction_days:x,0])
    
x_test = np.array(x_test)
x_test = np.reshape(x_test , (x_test.shape[0], x_test.shape[1] ,1 ))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform((predicted_prices))

# plot

plt.plot(actual_prices , color="black" , label=f"Actual{company} Price")
plt.plot(predicted_prices , color="green" ,label=f"Predicted{company} Price")
plt.title(f"{company}Share Price")
plt.xlabel('time')
plt.ylabel(f'{company} share price')
plt.legend()
plt.show()



# predict Next Day

real_data = [model_input[len(model_input)+1 - prediction_days:len(model_input+1) , 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data , (real_data.shape[0] , real_data.shape[1], 1))

# print(scaler.inverse_transform(real_data))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f'Prediction of next day{prediction}')