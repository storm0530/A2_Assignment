# A2_Assignment

## This repository is related to a course assignment.

### **I performed a multiple regression analysis on the missing BMI data to complement it.**

The implementation code and the data I used are as follows:

* **</span>.vscode</span>** : A directory for VSCode editing

* **</span>README.md</span>** : Overview of this repository

* **Missing_data_completion.py** : Code that implements a method to complement BMI values using multiple regression analysis.

* **blood_test_data.csv** : Blood test data (with missing data)

* **blood_test_data_edited.csv** : Numeric converted blood test data (with missing values)

* **result** : A directory containing the results of multiple regression analysis. The subdirectories are as follows:
    
    * **not_use_dummy** : Results of the analysis without dummy variables
    
    * **use_dummy** : Results of the analysis with dummy variables

    The data names in the subdirectories are as follows:
    
    * The header names are written as (Name,Gender,Age,Position,Drink,Smoke,Total_Cholesterol) = (N,G,A,P,D,S,T). 
    
    * For file names, the files are named for each header excluded as an explanatory variable. (e.g. "result_without_**NP**.txt" is analyzed using the values of Gender, Age, Drink, Smoke, and Total_Cholesterol **other than Name and Position** as explanatory variables.)