from sklearn.datasets import load_diabetes 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = load_diabetes(as_frame=True)
y = pd.DataFrame(df["target"])
x = pd.DataFrame(df["data"])

plt.scatter(x["age"], y, color="red", label="age")
plt.scatter(x["sex"], y, color="blue", label="sex")
plt.scatter(x["bmi"], y, color="green", label="bmi")

plt.legend(loc="upper right")

# plt.show()

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=.2)

x_data = x_train.to_numpy()
y_data = y_train.to_numpy()
x_test = x_test

model = LinearRegression()
model.fit(x_data, y_data)
y_predict = model.predict(x_test)

print(y_test, y_predict)

new_df = pd.DataFrame({"actual":y_test.to_numpy().flatten(), "predicted":y_predict.flatten()})
new_df.plot(kind="bar")
plt.show()




