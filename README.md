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


import matplotlib.pyplot as plt    
%matplotlib inline                  

import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
data=pd.read_csv(r"C:\Users\L390 Yoga\Downloads\Employee.csv")
data.head()

data.info()

data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x = data[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours",
          "time_spend_company", "Work_accident", "promotion_last_5years", "salary"]]
x.head() #no departments and no left

y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
plt.figure(figsize=(15,10))
plot_tree(dt,
          feature_names=x.columns,
          class_names=['stayed','left'],
          filled=True)

plt.show()

## Output:

<img width="1318" height="364" alt="image" src="https://github.com/user-attachments/assets/7b290aef-a3cb-4b9f-b59c-ea34c6b8603f" />

<img width="1290" height="373" alt="image" src="https://github.com/user-attachments/assets/f104f10f-dba8-4ac5-b7ce-eb9ea2aca927" />


<img width="1285" height="267" alt="image" src="https://github.com/user-attachments/assets/ff381d6b-7e53-4c9b-8d88-6770bc0c3f14" />



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
