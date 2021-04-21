# Multiple regression analysis code

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import Linear regression model (線形回帰モデル)
from sklearn.linear_model import LinearRegression

# Convert strings to numbers as appropriate
def numeric_conversion(str):
    if str == "Male" or str == "Professor" or str == "No":
        str = 0
    elif str == "Female" or str == "Associate Professor" or str == "Occasionally" or str == "Yes":
        str = 1
    elif str == "Assistant Professor" or str == "Often":
        str = 2
    return str
    
# setting header
header = ["Name","Gender","Age","Position","BMI","Drink","Smoke","Total_Cholesterol(mg/dL)"]

# load from csv
df = pd.read_csv("blood_test_data.csv")

# Numeric_conversion
for i in range(len(df)):
    for j in header:
        df.at[i,j] = numeric_conversion(df.at[i,j])

# Save conversion data
df.to_csv("./blood_test_data_edited.csv")

# Separate the "Ada" data for training and the other data for testing
df_train = df.iloc[1:,:]
df_test = df.iloc[:1,:]

# response variable(目的変数)(Y)，explanatory variable(説明変数)(X)
Y_train = np.array(df_train['BMI'])
X_train = np.array(df_train[["Gender","Age","Position","Drink","Smoke","Total_Cholesterol(mg/dL)"]])

# Specifying a regression model (回帰モデルの指定)
model = LinearRegression()

# Model fitting (学習開始)
model.fit(X_train, Y_train)

# Prediction BMI
X_test = np.array(df_test[["Gender","Age","Position","Drink","Smoke","Total_Cholesterol(mg/dL)"]])
Y_pred = model.predict(X_test)

print(f"According to the multiple regression analysis by Arashi Fukui,\n\
    Ada's BMI is about {Y_pred[0]}.")
# Ada's BMI is about 24.5687...