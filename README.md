# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Read the given dataset. 
2.Fitting the dataset into the training set and test set. 
3. Applying the feature scaling method.
4.Fitting the logistic regression into the training set.
5.Prediction of the test and result.
6.Making the confusion matrix.
7.Visualizing the training set results.
```
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: REXLIN R
RegisterNumber:212222220034
*/
```
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datasets = pd.read_csv("/content/Social_Network_Ads (1).csv")
X = datasets.iloc[:,[2,3]].values
Y = datasets.iloc[:,[4]].values

from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_X

X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.fit_transform(X_Test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_Train, Y_Train)

Y_Pred = classifier.predict(X_Test)
Y_Pred

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_Test, Y_Pred)
cm

from sklearn import metrics
accuracy = metrics.accuracy_score(Y_Test, Y_Pred)
accuracy

recall_sensitivity = metrics.recall_score(Y_Test, Y_Pred, pos_label = 1)
recall_specificity = metrics.recall_score(Y_Test, Y_Pred, pos_label = 0)
recall_sensitivity, recall_specificity

from matplotlib.colors import ListedColormap
X_Set, Y_Set = X_Train, Y_Train
X1,X2 = np.meshgrid(np.arange(start = X_Set[:,0].min()-1, stop = X_Set[:,0].max()+1, step = 0.01), 
                    np.arange(start = X_Set[:,1].min()-1, stop = X_Set[:,1].max()+1, step = 0.01))

plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),
X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red','green')))

plt.xlim(X1.min(), X2.max())
plt.ylim(X1.min(), X2.max())
for i,j in enumerate(np.unique(Y_Set)):
    plt.scatter(X_Set[Y_Set == j,0],X_Set[Y_Set==j,1],
                c=ListedColormap(('red','green'))(i),label=j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.label('Estimated Salary')
plt.legend()
plt.show()
```

## Output:

## Prediction of Test Result:
![image](https://github.com/rexlinrajan2004/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119406566/cf0166ea-03a8-4dcb-87d5-a6822c809b1f)

## Confusion Matrix:
![image](https://github.com/rexlinrajan2004/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119406566/275fb7af-ffda-4447-9c70-aea218a52eb7)

## Accuracy:
![image](https://github.com/rexlinrajan2004/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119406566/389422dc-5d32-40b8-924d-4558284a399d)

## Recalling Sensitivity and Specificity:
![image](https://github.com/rexlinrajan2004/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119406566/321e23bf-d260-4162-8c18-2778da3624b3)

## Visulaizing Training set Result:
![image](https://github.com/rexlinrajan2004/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119406566/a5db27de-15db-4f82-b6c2-3e2ba707ee5c)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

