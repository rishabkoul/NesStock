#Import the libraries
import os
import math
from dateutil.relativedelta import relativedelta
#import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from tensorflow.keras.models import load_model
from nsepy import get_history
from datetime import date
import datetime
import streamlit as st

from PIL import Image 
import tensorflow.keras.backend as tb
#tb._SYMBOLIC_SCOPE.value = True

def main():
    
    image = Image.open('sm.jpeg')
    st.image(image,use_column_width=True)

    st.title("NSE Real-Time Stocks Analysis and Predictions")
    
    st.header("Select the stock and check its next day predicted value")
    
    st.subheader("This study is mainly confined to the s tock market behavior and is \
                intended to devise certain techniques for investors to make reasonable\
                returns from investments .")
    
    st.subheader("Though there were a number of studies , which\
                deal with analysis of stock price behaviours , the use of control chart\
                techniques and fai lure time analysis would be new to the investors. The\
                concept of stock price elast icity,\
                introduced in this study, will be a good\
                tool to measure the sensitivity of stock price movements.")
    
    st.subheader("In this study, \
                 Predictions for the close price is suggested for the National Stock Exchange index,\
                Nifty,\
                based on Long Short Term Based (LSTM)\
                method.") 
    
    st.subheader("We make predictions based on the last 30 days Closing price data\
                which we fetch from NSE India website in realtime.")
    
    st.markdown("Note: This is just a fun project, No one can predict the\
         stock market as of today because there are a\
        lot of factors which needs to be considered\
             before makaing any investments, especially in StockMarket.\
            So it is advisable now to indulge in any\
                 bad decisions based on the predictions shown here.")

    # Stock Section
    st.markdown("[Choose NSE tickers from this list Click here](https://indiancompanies.in/listed-companies-in-nse-with-symbol/)")
    choose_stock = st.text_input("Enter NSE Stock ticker")

    if(choose_stock != ""):

        df = get_history(symbol=choose_stock, start=date.today()-relativedelta(years=10), end=date.today())
        df['Date'] = df.index

        st.header(choose_stock+" NSE Last 5 Days DataFrame:")
        
        # Insert Check-Box to show the snippet of the data.
        
        st.subheader("Showing raw data---->>>")	
        st.dataframe(df.tail())
        
        #Create a new dataframe with only the 'feature' column
        data = df.filter(['Close'])
        #Converting the dataframe to a numpy array
        dataset = data.values
        #Get /Compute the number of rows to train the model on
        training_data_len = math.ceil( len(dataset) *.75)

        #Scale the all of the data to be values between 0 and 1 
        scaler = MinMaxScaler(feature_range=(0, 1)) 
        scaled_data = scaler.fit_transform(dataset)

        #Create the scaled training data set 
        train_data = scaled_data[0:training_data_len  , : ]

        #Split the data into x_train and y_train data sets
        x_train=[]
        y_train = []
        for i in range(30, len(train_data)):
            x_train.append(train_data[i-30:i,0])
            y_train.append(train_data[i,0])
    
        #Convert x_train and y_train to numpy arrays
        x_train, y_train = np.array(x_train), np.array(y_train)

        #Reshape the data into the shape accepted by the LSTM
        x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

        #Build the LSTM network model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dense(units=25))
        model.add(Dense(units=1))

        #Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        #Train the model
        with st.spinner('Training may take time...'):
            model.fit(x_train, y_train, batch_size=50, epochs=100)
        st.success('Done!')

        #Test data set
        test_data = scaled_data[training_data_len - 30: , : ]

        #Create the x_test and y_test data sets
        x_test = []
        y_test =  dataset[training_data_len : , : ] #Get all of the rows from index 1603 to the rest and all of the columns (in this case it's only column 'Close'), so 2003 - 1603 = 400 rows of data
        for i in range(30,len(test_data)):
            x_test.append(test_data[i-30:i,0])
    
        #Convert x_test to a numpy array 
        x_test = np.array(x_test)

        #Reshape the data into the shape accepted by the LSTM
        x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

        #Getting the models predicted price values
        predictions = model.predict(x_test) 
        predictions = scaler.inverse_transform(predictions)#Undo scaling

        #Calculate/Get the value of RMSE
        rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
        rmse

        #Plot/Create the data for the graph
        train = data[:training_data_len]
        valid = data[training_data_len:]
        valid['Predictions'] = predictions

        #Visualize the data
        plt.figure(figsize=(16,8))
        plt.title('Model')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price', fontsize=18)
        plt.plot(train['Close'])
        plt.plot(valid[['Close', 'Predictions']])
        plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
        plt.savefig("mygraph.png")

        image = Image.open('mygraph.png')
        st.image(image,use_column_width=True)

        #Show the valid and predicted prices
        valid[-5:]

        #Create a new dataframe
        new_df = df.filter(['Close'])
        #Scale the all of the data to be values between 0 and 1 
        scaler = MinMaxScaler(feature_range=(0, 1)) 
        scaled_data = scaler.fit_transform(new_df)
        #Get teh last 30 day closing price 
        last_30_days = new_df[-30:].values
        #Scale the data to be values between 0 and 1
        last_30_days_scaled = scaler.transform(last_30_days)
        #Create an empty list
        X_test = []
        #Append teh past 1 days
        X_test.append(last_30_days_scaled)
        #Convert the X_test data set to a numpy array
        X_test = np.array(X_test)
        #Reshape the data
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        #Get the predicted scaled price
        pred_price = model.predict(X_test)
        #undo the scaling 
        pred_price = scaler.inverse_transform(pred_price)

        # next day
        NextDay_Date = datetime.date.today() + datetime.timedelta(days=1)

        st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
        st.markdown(pred_price)

        ##visualizations
        st.subheader("Close Price VS Date Interactive chart for analysis:")
        st.area_chart(df['Close'])

        st.subheader("Line chart of Open and Close for analysis:")
        st.area_chart(df[['Open','Close']])

        st.subheader("Line chart of High and Low for analysis:")
        st.line_chart(df[['High','Low']])

if __name__ == '__main__':

    main()
