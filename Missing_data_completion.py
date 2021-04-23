# Multiple regression analysis code

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import statsmodels.api as sm
import statsmodels.formula.api as smf

DUMMY_FLAG = 1 # 1: Use dummy variables, 0: Don't use

# select header for explanatory variable
HEADER_FOR_ANALYZE = ["Gender","Age","Drink","Total_Cholesterol(mg/dL)"]
# all header is ["Name","Gender","Age","Position","BMI","Drink","Smoke","Total_Cholesterol(mg/dL)"]

# Function to find the response variable
def predict_y(X, coef, header):
    # X(dictionary type) : explanatory variable
    # coef : coefficients of the regression equation
    # header : header for explanatory variable
    
    y = coef["const"]
    for i in header:
        y += coef[i] * X[i]

    return y




# Dictionary for conversion strings to numbers as appropriate
numeric_dict = {
    "Female":0,"Male":1,"No":0,"Yes":1,
    "Occasionally":1,"Often":2,
    "Assistant Professor":1,
    "Associate Professor":2,
    "Professor":3
}

# load from csv
df = pd.read_csv("blood_test_data.csv")
pd.set_option('display.max_rows', None)

# Numeric_conversion
for i in range(len(df)):
    for j in HEADER_FOR_ANALYZE:
        if df.at[i,j] in numeric_dict:
            df.at[i,j] = numeric_dict[df.at[i,j]]

# save data after conversion
df.to_csv("./blood_test_data_edited.csv")

# Extract only header values used for analysis
if DUMMY_FLAG == 1:
    # make dummy variables
    HEADER_FOR_ANALYZE.append("BMI")
    df = pd.get_dummies(df[HEADER_FOR_ANALYZE]).astype(float)
    
    # add dummy to header selected for explanatory variable
    HEADER_FOR_ANALYZE = list(df)
else:
    HEADER_FOR_ANALYZE.append("BMI")
    df = df[HEADER_FOR_ANALYZE].astype(float)

HEADER_FOR_ANALYZE.remove("BMI")


# separate the "Ada" data for training and the other data for testing
df_train = df.iloc[1:,:]
df_test = df.iloc[:1,:]

# X: explanatory variables, Y: response variable
X_train = (df_train[HEADER_FOR_ANALYZE]).astype(float)
X_test = (df_test[HEADER_FOR_ANALYZE]).astype(float)
Y_train = (df_train['BMI']).astype(float)

# add a constant term to the regression equation
X_train = sm.add_constant(X_train)

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
predict = predict_y(X_test,result_coef,HEADER_FOR_ANALYZE)

# save results of multiple regression analysis and prediction
with open('result.txt', 'w') as f:
    print(f"{result.summary()}\n\n",file=f)
    print(f"According to the multiple regression analysis by Arashi Fukui,\n\
Ada's BMI is about {predict}.",file=f)
# Ada's BMI is about 24.4609...