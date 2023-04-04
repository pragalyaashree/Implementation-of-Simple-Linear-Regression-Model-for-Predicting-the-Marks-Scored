# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Pandas as pd & Import numpy as np

2.Calulating The y_pred & y_test

3.Find the graph for Training set & Test Set

4.Find the values of MSE,MSA,RMSE

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: R.K Pragalyaa shree
RegisterNumber:  212221040125
*/
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv("/content/student_scores.csv")
#displaying the content in datafile
df.head()

df.tail()

#segregating data to variables
x=df.iloc[:,:-1].values
x

y=df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression 
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

y_pred

y_test


plt.scatter(x_train,y_train,color="orange")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()


plt.scatter(x_test,y_test,color="purple")
plt.plot(x_train,regressor.predict(x_train),color="green")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print("MSE= ",mse)

mae=mean_absolute_error(y_test,y_pred)
print("MAE = ",mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)




## Output:
## df.head():

![ml 1](https://user-images.githubusercontent.com/128135934/229762238-b5072d53-8225-40e1-9c20-2ad66d48d3ad.png)

## df.tail():

![ml2](https://user-images.githubusercontent.com/128135934/229762657-331b2fb3-f887-4bbc-b9c8-1ecbd3f25315.png)

## Array of X:

![ml3](https://user-images.githubusercontent.com/128135934/229763055-9d7a86d9-0970-4a6e-951c-e8981a121fe9.png)

## Array of Y:

![ml4](https://user-images.githubusercontent.com/128135934/229763419-aae12569-00ec-4f7e-b13d-116cef93a674.png)

## Y_Pred:

![ml5](https://user-images.githubusercontent.com/128135934/229763710-caa73991-7296-4b5d-a2f5-b3f1f3ff7c51.png)

## y_test:

![ml6](https://user-images.githubusercontent.com/128135934/229763879-2ea4b808-48c8-4c4e-9d3b-df947cfbbb9e.png)

## Training Set:

![ml7](https://user-images.githubusercontent.com/128135934/229764101-0f891d2b-a5ad-46f6-931f-414016096b7d.png)

## Test Set:

![ml8](https://user-images.githubusercontent.com/128135934/229764272-5f422758-a4ab-4341-8506-9455dfa18945.png)

## Values of MSE,MAE,RMSE:

![ml9](https://user-images.githubusercontent.com/128135934/229764492-31b6125b-9ca5-4333-ad63-cfacb45f5f7a.png)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.


