import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('Advertising.csv')

x = df['TV'].to_numpy().reshape(-1,1)
y = df['sales'].to_numpy()

plt.scatter(x,y, color="red", label="Actual")

model = LinearRegression()
model.fit(x,y)
y_pred  = model.predict(x)


plt.plot(x,y_pred, label="Predicted", color="green")
plt.xlabel('TV')
plt.ylabel('Sales')


print("error", mean_squared_error(y, y_pred))
print("accuracy", r2_score(y, y_pred))

plt.legend()
plt.show()
