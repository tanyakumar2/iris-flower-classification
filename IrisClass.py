#import packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


columns = ['Sepal Length', 'Sepal width', 'Petal length', 'Petal width', 'Class_labels']

#upload data
df = pd.read_csv('iris.data', names=columns)

df.head()

#stats about the data
df.describe()

#visualize data set
sns.pairplot(df, hue='Class_labels')
plt.show() 

#seperate data and results
data = df.values
X = data[:,0:4]
Y = data[:,4]

#calculate average for each type
Y_Data = np.array([np.average(X[:, i][Y==j].astype('float32')) for i in range (X.shape[1]) for j in (np.unique(Y))])
Y_Data_reshaped = Y_Data.reshape(4, 3)
Y_Data_reshaped = np.swapaxes(Y_Data_reshaped, 0, 1)
X_axis = np.arange(len(columns)-1)
width = 0.25

#plot the average
plt.bar(X_axis, Y_Data_reshaped[0], width, label = 'Setosa')
plt.bar(X_axis+width, Y_Data_reshaped[1], width, label = 'Versicolour')
plt.bar(X_axis+width*2, Y_Data_reshaped[2], width, label = 'Virginica')
plt.xticks(X_axis, columns[:4])
plt.xlabel("Features")
plt.ylabel("Value in cm.")
plt.legend(bbox_to_anchor=(1.3,1))


#split the data to train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

#support vector machine algorithm
from sklearn.svm import SVC
svn = SVC()
svn.fit(X_train, y_train)

#predict from the test dataset
predictions = svn.predict(X_test)

#calculate the accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))


#classification report
from sklearn.metrics import classification_report
classification_report(y_test, predictions)

#measurements of iris petals, input vector
X_new = np.array([[3, 2, 1, 0.2], [  5.3, 2.5, 4.6, 1.9 ], [  4.9, 2.2, 3.8, 1.1 ]])

#prediction of species from input vector
prediction = svn.predict(X_new)
print("Prediction of Species: {}".format(prediction))





