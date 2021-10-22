import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

def model(bedrooms,bathrooms,sqft_living,sqft_lot,floors,waterfront,view,condition):
    data = pd.read_csv('data.csv')
    data.drop(['date','street','city','statezip','country', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated'], axis=1, inplace=True)
    X = data.drop('price', axis=True)
    Y = data['price']

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)

    model = LinearRegression()
    model.fit(X_train,Y_train)

    def print_evaluate(true, predicted):  
        mae = metrics.mean_absolute_error(true, predicted)
        mse = metrics.mean_squared_error(true, predicted)
        rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
        r2_square = metrics.r2_score(true, predicted)
        print('MAE:', mae)
        print('MSE:', mse)
        print('RMSE:', rmse)
        print('R2 Square', r2_square)
        print('__________________________________')

    # y_pred = model.predict(X_test)
    # print_evaluate(Y_test, y_pred);

    print(X_train.info())
    data_income = np.array([(bedrooms,bathrooms,sqft_living,sqft_lot,floors,waterfront,view,condition)])
    data_pred = model.predict(data_income)
    return int(data_pred)

st.title("House Price Prediction App")
st.subheader('')

#Text Input
bedrooms = st.text_input("Number of Bedrooms", '')
bathrooms = st.text_input("Number of Bathrooms", '')
sqft_living = st.text_input("Number of Sqft Living", '')
sqft_lot = st.text_input("Number of Sqft Lot", '')
floors = st.text_input("Number of Floors", '')
waterfront = st.text_input("Number of waterfront", '')
view = st.text_input("Number of View", '')
condition = st.text_input("Number of Condition", '')

if st.button("Submit"):
    result = model(bedrooms,bathrooms,sqft_living,sqft_lot,floors,waterfront,view,condition)
    st.success(result)

# model(2,3,1300,2400,2,0,1,3)