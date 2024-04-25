# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm



1. Use the standard libraries in python for finding linear regression.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Predict the values of array.
5. Calculate the accuracy, confusion and classification report b importing the required modules from sklearn.
6. Obtain the graph.









## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by:R.Subhashri
 
RegisterNumber: 212223230219



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data = np.loadtxt("ex2data1.txt",delimiter=",")
X = data[:,[0,1]]
Y = data[:,2]

X[:5]

Y[:5]

# VISUALIZING THE DATA
plt.figure()
plt.scatter(X[Y== 1][:, 0], X[Y==1][:,1],label="Admitted")
plt.scatter(X[Y==0][:,0],X[Y==0][:,1],label="Not admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction(theta, X, Y):
    h = sigmoid(np.dot(X, theta))
    J = -(np.dot(Y, np.log(h)) + np.dot(1-Y,np.log(1-h))) / X.shape[0]
    grad = np.dot(X.T, h-Y)/X.shape[0]
    return J,grad

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta = np.array([0,0,0])
J,grad = costFunction(theta,X_train,Y)
print(J)
print(grad)

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta = np.array([-24,0.2,0.2])
J,grad = costFunction(theta,X_train,Y)
print(J)
print(grad)

def cost(theta,X,Y):
  h=sigmoid(np.dot(X,theta))
  J=-(np.dot(Y,np.log(h))+np.dot(1-Y,np.log(1-h)))/X.shape[0]
  return J

def gradient(theta,X,Y):
  h=sigmoid(np.dot(X,theta))
  grad=np.dot(X.T,h-Y)/X.shape[0]
  return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,Y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,Y):
    X_min , X_max = X[:, 0].min() - 1,X[:,0].max() + 1
    Y_min , Y_max = X[:, 1].min() - 1,X[:,1].max() + 1
    XX,YY = np.meshgrid(np.arange(X_min,X_max,0.1),
                        np.arange(Y_min,Y_max,0.1))
    X_plot = np.c_[XX.ravel(), YY.ravel()]
    X_plot = np.hsatck((np.ones((X_plot.shape[0],1)),X_plot))
    Y_plot = np.dot(X_plot, theta).reshape(XX.shape)
    plt.figure()
    plt.scatter(X[Y==1][:,0],X[Y==1][:,1],label='Admitted')
    plt.scatter(X[Y==1][:,0],X[Y==1][:,1],label='Not admitted')
    plt.contour(XX,YY,Y_plot,levels=[0])
    plt.Xlabel("Exam 1 score")
    plt.Ylabel("Exam 2 score")
    plt.legend()
    plt.show()

print("Decision boundary-graph for exam score:")
plotDecisionBoundary(res.x,X,Y)


prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta, X):
  X_train=np.hstack((np.ones((X.shape[0],1)),X))
  prob=sigmoid(np.dot(X_train,theta))
  return (prob >= 0.5).astype(int)

np.mean(predict(res.x,X)==y)








 
*/
```

## Output:

### Array value of X:

![Screenshot 2024-04-24 203357](https://github.com/SubhashriRavichandran10/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145743413/eaeb1703-a641-40e8-aa34-6dc943405fa2)



### Array value of Y:



![Screenshot 2024-04-24 203403](https://github.com/SubhashriRavichandran10/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145743413/b1965e53-7ff5-482d-acdf-defebc8205a0)



### Exam 1-Score graph:


![Screenshot 2024-04-24 203412](https://github.com/SubhashriRavichandran10/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145743413/3366f506-0be4-4dd7-ae34-5c0ee2c7f6e3)




### Sigmoid function graph:


![Screenshot 2024-04-24 203420](https://github.com/SubhashriRavichandran10/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145743413/fa3ec656-5e2c-4b09-975a-f4e56a4584c6)



### X_Train_grad value:


![Screenshot 2024-04-24 203426](https://github.com/SubhashriRavichandran10/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145743413/8e6cb680-db1e-4fcb-82c7-9ca90dbfd521)


### Y_Train_grad value:

![Screenshot 2024-04-24 203431](https://github.com/SubhashriRavichandran10/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145743413/2fa24317-621e-41d2-a63a-7a8c15c7f90f)



### Print res.X:

![Screenshot 2024-04-24 203437](https://github.com/SubhashriRavichandran10/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145743413/0189a4f2-d95b-48e5-861b-d671c9654e89)




### Decision boundary-gragh for exam score:

![Screenshot 2024-04-24 203445](https://github.com/SubhashriRavichandran10/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145743413/6a0016a1-9dbd-4018-bf85-9fa1e5afa761)



### Probability value:




![Screenshot 2024-04-24 203449](https://github.com/SubhashriRavichandran10/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145743413/e0421d2b-338c-4530-bab7-6885c2c727d2)




### Prediction value of mean:

![Screenshot 2024-04-24 203454](https://github.com/SubhashriRavichandran10/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145743413/e57a1c23-58d7-408b-bffe-964ec7fc1117)





## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

