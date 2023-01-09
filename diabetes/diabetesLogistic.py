import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
df = pd.read_csv('diabetes.csv')

print(df.head())

y = df["Outcome"]
x = df.iloc[:,0:8]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=44)

model = LogisticRegression(max_iter=400)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
cf_matrix = metrics.confusion_matrix(y_test, y_pred)

print(accuracy)
print(cf_matrix )