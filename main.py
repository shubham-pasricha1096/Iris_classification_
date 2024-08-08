import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('C:\\Users\\shubh\\OneDrive\\Desktop\\archive\\IRIS.csv')

print(df.head())

print(df.describe())

print(df.info())

print(df.shape)

print(df['species'].value_counts())

df['sepal_length'].hist()
# plt.show()

df['sepal_width'].hist()
# plt.show()

df['petal_length'].hist()
# plt.show()

df['petal_width'].hist()
# plt.show()

colors = ['red','orange','blue']
species = ['Iris-setosa', 'Iris-versicolor','Iris-virginica']

for i in range(3):
    x = df[df['species'] == species[i]]
    plt.scatter(x['sepal_length'],x['sepal_width'], c = colors[i],label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()
# plt.show()

for i in range(3):
    x = df[df['species'] == species[i]]
    plt.scatter(x['petal_length'],x['petal_width'], c = colors[i],label=species[i])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend()
# plt.show()

for i in range(3):
    x = df[df['species'] == species[i]]
    plt.scatter(x['sepal_length'],x['petal_length'], c = colors[i],label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend()
# plt.show()

for i in range(3):
    x = df[df['species'] == species[i]]
    plt.scatter(x['sepal_width'],x['petal_width'], c = colors[i],label=species[i])
plt.xlabel("Sepal Width")
plt.ylabel("Petal Width")
plt.legend()
# plt.show()


# df = df.drop(columns=['species'])
# print(df.corr())

# corr = df.corr()
# fig, ax = plt.subplots(figsize = (5,4))
# sns.heatmap(corr,annot=True,ax=ax,cmap='coolwarm')
# plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle

X = df.drop(columns=['species'])
Y = df['species']
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.30)

model = LogisticRegression()
model.fit(x_train,y_train)
print("Logical Regression Accuracy: ",model.score(x_test,y_test)*100)

print(model.fit(x_train.values,y_train.values))

print("Accuracy: ",model.score(x_test,y_test)*100)

model = KNeighborsClassifier()
model.fit(x_train.values,y_train.values)
print("K-nearest neighbors Accuracy: ",model.score(x_test,y_test)*100)

print(model.fit(x_train.values,y_train.values))

print("Accuracy: ",model.score(x_test,y_test)*100)

model = DecisionTreeClassifier()
model.fit(x_train,y_train)
print("Decision Tree Accuracy: ",model.score(x_test,y_test)*100)

print(model.fit(x_train.values,y_train.values))

print("Accuracy: ",model.score(x_test,y_test)*100)

import pickle
filename = 'saved_model.sav'
pickle.dump(model,open(filename,'wb'))

load_model = pickle.load(open(filename,'rb'))

print(load_model.predict([[4,3,1,5]]))



