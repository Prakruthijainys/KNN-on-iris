# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 11:09:35 2021

@author: User
"""
#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read dataset
dataset = pd.read_csv('Iris.csv', delimiter=',')
print(dataset.shape)

# split into input (X) and output (y) variables
x = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values

#split the dataset into test and train data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y,test_size = 0.25,random_state =0)
print ('Train set:', x_train.shape,  y_train.shape)
print ('Test set:', x_test.shape,  y_test.shape)

#standardisation
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

#apply knn classifier 
from sklearn.neighbors import KNeighborsClassifier
#train the model
for k in range(1,10):
    knn = KNeighborsClassifier(n_neighbors = k).fit(x_train,y_train)
    print(knn)
    #predict the test data
    y_pred = knn.predict(x_test)
    #calculate the accuracy
    from sklearn import metrics
    print("Train set Accuracy: " ,metrics.accuracy_score(y_train,knn.predict(x_train)))
    print("Test set Accuracy: ",metrics.accuracy_score(y_test, y_pred)) 
      
#plot the curve
no_neighbors = np.arange(1,10)
train_accuracy = np.empty(len(no_neighbors))
test_accuracy = np.empty(len(no_neighbors))
plt.figure(figsize=(20,20))
plt.grid(color = 'grey', linestyle =':',linewidth = 1)
plt.plot(no_neighbors ,test_accuracy,color = 'green',marker = 'o',label = 'testing accuracy')
plt.plot(no_neighbors ,train_accuracy,color='red',marker ='o',label = 'training accuracy')
plt.xlabel('value of k')
plt.ylabel('accuracy')
plt.show()




    