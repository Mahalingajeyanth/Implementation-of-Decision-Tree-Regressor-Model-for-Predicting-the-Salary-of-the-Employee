# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the Salary dataset using pandas.

2. Encode the categorical “Position” column using LabelEncoder.

3. Split the data into training and testing sets.

4. Train & Predict using DecisionTreeRegressor and visualize the tree. 

## Program:


### Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
---
```
Developed by: MAHALINGA JEYANTH V
RegisterNumber: 212224220057
```
---
```
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
```

```py
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
```

```py
x=data[["Position","Level"]]
x.head()
y=data["Salary"]
y.head()

```

```py
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
```

```py
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
y_pred
from sklearn.metrics import r2_score
r2=r2_score(y_test,y_pred)
print("R2 Score = ",r2)
```

```py
dt.predict([[5,6]])
```

```py
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(12,8))
plot_tree(
    dt,
    feature_names=["Position", "Level"],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree for Salary Prediction", fontsize=14)
plt.show()
```


## Output:

<img width="928" height="553" alt="image" src="https://github.com/user-attachments/assets/7280e302-48fe-4006-a6a8-6fdb636150d1" />


<img width="870" height="329" alt="image" src="https://github.com/user-attachments/assets/fa662034-ee8f-4921-bcbc-c55bcb267d15" />


<img width="890" height="378" alt="image" src="https://github.com/user-attachments/assets/457ce189-8623-4684-b98b-a124aff53533" />


<img width="867" height="28" alt="image" src="https://github.com/user-attachments/assets/28babdea-d48e-48bb-8dff-e0db40476618" />


<img width="879" height="30" alt="image" src="https://github.com/user-attachments/assets/a780138d-6faf-4f1e-b5ec-23bd4d55f941" />


<img width="1364" height="946" alt="image" src="https://github.com/user-attachments/assets/e0800105-aa47-45f7-95c2-8ebefd59cb29" />



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
