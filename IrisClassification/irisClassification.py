import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
df = pd.read_csv('Iris.csv')


y = df["Species"]
x = df.iloc[:,1:5]

print(len(y), len(x))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)

model = LogisticRegression(max_iter=400)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
matrix = metrics.confusion_matrix(y_test, y_pred)



print(accuracy)
print(matrix)

