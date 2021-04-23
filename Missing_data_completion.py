# Multiple regression analysis code

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import statsmodels.api as sm
import statsmodels.formula.api as smf

DUMMY_FLAG = 0 # 1: Use dummy variables, 0: Don't use
POSITION_FLAG = 0 # 1: Use Position value, 0: Don't use

# Function to find the response variable
def predict_y(X, coef, header):
    # X(dictionary type) : explanatory variable
    # coef : coefficients of the regression equation
    # header : headers for analyze
    
    y = coef["const"]
    for i in header:
        y += coef[i] * X[i]

    return y



# Dictionary for conversion strings to numbers as appropriate
if POSITION_FLAG == 0:
    numeric_dict = {"Female":0,"Male":1,"No":0,"Yes":1,"Occasionally":1,"Often":2}
else:
    numeric_dict = {"Female":0,"Male":1,"No":0,"Yes":1,"Occasionally":1,"Often":2,
    "Assistant Professor":1,"Associate Professor":2,"Professor":3
    }

# setting header list
header = ["Name","Gender","Age","Position","BMI","Drink","Smoke","Total_Cholesterol(mg/dL)"]


# load from csv
df = pd.read_csv("blood_test_data.csv")
pd.set_option('display.max_rows', None)

# Numeric_conversion
for i in range(len(df)):
    for j in header:
        if df.at[i,j] in numeric_dict:
            df.at[i,j] = numeric_dict[df.at[i,j]]

# save data after conversion
df.to_csv("./blood_test_data_edited.csv")

# Not use dummy variables
if DUMMY_FLAG == 0:
    
    # Not use position value
    if POSITION_FLAG == 0:
        header_for_analyze = ["Gender","Age","Drink","Smoke","Total_Cholesterol(mg/dL)"]
    
    # Use position value
    else:
        header_for_analyze = ["Gender","Age","Position","Drink","Smoke","Total_Cholesterol(mg/dL)"]

# Use dummy variables
else:
    df = pd.get_dummies(df[["Gender","Age","Position","BMI","Drink","Smoke","Total_Cholesterol(mg/dL)"]])
    
    # Not use position value
    if POSITION_FLAG == 0:
        header_for_analyze = ["Gender_0","Gender_1","Age","Drink_0","Drink_1","Drink_2","Smoke_0","Smoke_1","Total_Cholesterol(mg/dL)"]
        
    # Use position value
    else:
        header_for_analyze = ["Gender_0","Gender_1","Age","Position_1","Position_2","Position_3","Drink_0","Drink_1","Drink_2","Smoke_0","Smoke_1","Total_Cholesterol(mg/dL)"]


# separate the "Ada" data for training and the other data for testing
df_train = df.iloc[1:,:]
df_test = df.iloc[:1,:]

# Extract only header values used for analysis
x_train = (df_train[header_for_analyze]).astype(float)
X_test = (df_test[header_for_analyze]).astype(float)

Y_train = (df_train['BMI']).astype(float)

# add a constant term to the regression equation
X_train = sm.add_constant(x_train)

# transform into dictionary type
X_test = X_test.to_dict()
X_test = {i: X_test[i][0] for i in X_test}

# Modeling by Least Squares Method
model = sm.OLS(Y_train, X_train)
result = model.fit()


# display the results of multiple regression analysis
print(result.summary())

# get coefficients of the regression equation
result_coef = result.params

# Prediction of missing BMI values
predict = predict_y(X_test,result_coef,header_for_analyze)

# save results of multiple regression analysis and prediction
with open('result_summary.txt', 'w') as f:
    print(f"{result.summary()}\n\n",file=f)
    print(f"According to the multiple regression analysis by Arashi Fukui,\n\
Ada's BMI is about {predict}.",file=f)

# Ada's BMI is about 24.4609...