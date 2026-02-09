# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the libraries and read the data frame using pandas.
2.Calculate the null values present in the dataset and apply label encoder.
3.Determine test and training data set and apply decison tree regression in dataset.
4.Calculate Mean square error,data prediction and r2.

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: JANSI RANI A A
RegisterNumber:  212224040130


import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])


```

## Output:

## Data head:

<img width="319" height="245" alt="Screenshot (775)" src="https://github.com/user-attachments/assets/987cb16a-6bef-4c05-aaba-18a04b307fb0" />

## Data info:

<img width="670" height="256" alt="Screenshot (776)" src="https://github.com/user-attachments/assets/1765e029-1de6-4ba3-8012-0098ec8525bd" />

## isnull() sum():

<img width="179" height="104" alt="Screenshot (777)" src="https://github.com/user-attachments/assets/defd3abf-bb31-46cf-9981-73e4f0764cb0" />

## Mean squared error:

<img width="253" height="47" alt="Screenshot (778)" src="https://github.com/user-attachments/assets/598cfd06-092e-48e3-bf3f-f74861465e22" />

## r2 value:

<img width="299" height="39" alt="Screenshot (779)" src="https://github.com/user-attachments/assets/314774d9-3590-4f08-867d-5c43a4181b05" />

## data prediction:

<img width="230" height="37" alt="Screenshot (780)" src="https://github.com/user-attachments/assets/c57e5c64-b2ec-4943-9682-31171a9703b1" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
