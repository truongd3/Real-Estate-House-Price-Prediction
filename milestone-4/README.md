# Interpretable Gradient Boosting - Real Estate House Price Prediction (Milestone 4) --- Code Explanation

## Required Packages

Before implementing the algorithm into the webpage on HuggingFace, we **create ```requirements.txt```**. This file contains all the libraries and frameworks needed for the project.

```
ipython
lightgbm
streamlit
optuna
numpy
pandas
matplotlib
shap
```

To locally install all the packages in this file, type this command line on terminal:

> pip install -r requirements.txt

## Dataset

Next, **download ```train.csv```** from the link given in [milestone-2](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data?select=train.csv). This is the dataset we use to train the model. 

## Build the Webpage

Let's get into the main part - ```app.py```.

### Back-end

**Import** all the mentioned **packages** at the beginning:

```
import streamlit as  st # to host the page
import optuna # to tune with Optuna the hyperparameters of the LightGBM model
import numpy as np # to use numbers in Python
import pandas as pd # to upload data
import matplotlib
import matplotlib.pyplot as plt # to plot diagrams
import shap #
from sklearn.model_selection import train_test_split # to split columns from dataframe
import lightgbm as lgb
```

There are 79 features in total in the dataset but we don't have to use all of them. In this case, we can **select some *useful* features** and I got 17.

```
features = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'YearBuilt', 'BedroomAbvGr', 'FullBath', 'TotalBsmtSF', 'GarageArea', 'OverallCond',
'1stFlrSF', '2ndFlrSF', 'GrLivArea', 'Fireplaces', 'YrSold', 'YearRemodAdd', 'BsmtUnfSF']
```

**Data processing**:

```
df = pd.read_csv('train.csv') # upload dataset
for feature in features: # convert NaN to mean of columns
    df[feature].fillna(value=df[feature].mean(), inplace=True)

X = df[features] # inputs
y = df['SalePrice'] # correct output
```

Create data to **train and test using LGB**:

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
trainData = lgb.Dataset(X_train, label = y_train)
TestingData = lgb.Dataset(X_test, label = y_test, reference = trainData)

# parameters used for LightGBM with Optuna
Hyperparameters = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'max_depth': 10,
    'num_leaves': 100
}
 # train model
model_lgbm = lgb.train(Hyperparameters, trainData, valid_sets=TestingData, num_boost_round= 4000, early_stopping_rounds=50)
```

### Front-end

Basically, we have 3 main areas for the landing page: a header, 17 sliders to the left, and the area to display output to the right.

```
header = st.container()
columnWithSlider, col2 = st.columns((4, 10)) # 4/10 is the ratio of width between the two horizontal areas
```

- Simple **header**:

```
with header: # simple header
    st.title('House Price Prediction - Truong Dang')
```

- 17 **sliders** to the left:

  - ```st.slider``` requires 3 parameters: the title description, the minimum value, and the maximum value.
  
  - The 17 sliders variable are created for the 17 mentioned features respectively.
  
  - The minimum and maximum are consulted from ```train.csv```.

```
with columnWithSlider:
    features[0] = st.slider('What is the building class?', min_value = 20, max_value = 190) # MSSubClass
    features[1] = st.slider('What is the linear feet of street connected to property?', min_value = 21, max_value = 313) # LotFrontage
    features[2] = st.slider('What is the lot size in square foot?', min_value = 1300, max_value = 215245) # LotArea
    features[3] = st.slider('What is the overall material and finish quality?', min_value = 1, max_value = 10) # OverallQual
    features[4] = st.slider('What is the original construction date?', min_value = 1872, max_value = 2010) # YearBuilt
    features[5] = st.slider('How many bedrooms not including basement bedrooms?', min_value = 0, max_value = 8) # BedroomAbvGr
    features[6] = st.slider('How many full bathrooms?', min_value = 0, max_value = 3) # FullBath
    features[7] = st.slider('What is the total square feet of basement area?', min_value = 0, max_value = 6110) # TotalBsmtSF
    features[8] = st.slider('What is the size of garage in square feet?', min_value = 0, max_value = 1418) # GarageArea
    features[9] = st.slider('What is the overall condition rating?', min_value = 1, max_value = 9) # OverallCond
    OneFlrSF = st.slider('What is the 1st floor square feet?', min_value = 334, max_value = 4692)
    TwoFlrSF = st.slider('What is the 2nd floor square feet?', min_value = 0, max_value = 2065)
    features[12] = st.slider('What is the above grade (ground) living area square feet?', min_value = 334, max_value = 5642) # GrLivArea
    features[13] = st.slider('How many fireplaces?', min_value = 0, max_value = 3) # Fireplaces
    features[14] = st.slider('What is the year sold?', min_value = 2006, max_value = 2010) # YrSold
    features[15] = st.slider('In which year was it re-modelled?', min_value = 1950, max_value = 2010) # YearRemodAdd
    features[16] = st.slider('What is the Unfinished square feet of basement area??', min_value = 0, max_value = 2336) # BsmtUnfSF
```

- Area to **display output**:

    - Create a button where to activate the prediction using the model. After the algorithm is done, button is set to ```False``` again so it does not run forever. Which means, we need to make sure the program only runs when the button is clicked.
    - Create a hashmap of dataframe with key is the feature name and value is the ```features```'s elements.
    - We preduct the house price with given values through the sliders by using ***model_lgbm.predict*** and assign the value to variable ```y_predict```.
    - Finally, display the predicted output and the graphs of SHAP values.

```
with col2:
    st.text('Name: Truong Dang')
    st.text('Email: tdd4@njit.edu')
    st.text('Course: CS 301-102')
    st.text('NJIT ID: 31558941')
    
    button = st.button('Range of house price') # button created
    if (button): # if button is clicked, value is True
        data_df = {'MSSubClass': [features[0]],
                'LotFrontage': [features[1]],
                'LotArea': [features[2]],
                'OverallQual': [features[3]], 
                'YearBuilt': [features[4]],
                'BedroomAbvGr': [features[5]],
                'FullBath': [features[6]],
                'TotalBsmtSF': [features[7]],
                'GarageArea': [features[8]],
                'OverallCond': [features[9]],
                '1stFlrSF': [OneFlrSF],
                '2ndFlrSF': [TwoFlrSF],
                'GrLivArea': [features[12]],
                'Fireplaces': [features[13]],
                'YrSold': [features[14]],
                'YearRemodAdd': [features[15]],
                'BsmtUnfSF': [features[16]]
            }
        
        data_df = pd.DataFrame.from_dict(data_df)
        y_predict = model_lgbm.predict(data_df, predict_disable_shape_check=True)
        st.write("Predicted price: ", y_predict[0]) # Display the prediction
        # SHAP Beeswarm
        explainer = shap.TreeExplainer(model_lgbm, X_train)
        shap_values = explainer.shap_values(X_test)
        shap.initjs()
        shap.summary_plot(shap_values, X_test, plot_type="dot")
        plt.savefig('beeswarm.png', bbox_inches='tight')
        plt.close()
        st.image("beeswarm.png")
        # SHAP interaction
        shap.initjs()
        shap_interaction_values = shap.TreeExplainer(model_lgbm).shap_interaction_values(X.iloc[:,:])
        shap.summary_plot(shap_interaction_values, X.iloc[:,:])
        plt.savefig('interact.png', bbox_inches='tight')
        plt.close()
        st.image("interact.png")

        button = False 
```
