import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv('pop.csv')

_1pop = df[df["Country Name"] == "India"].reset_index()
_2pop = df[df["Country Name"] == "United States"].reset_index()
_3pop = df[df["Country Name"] == "China"].reset_index()


y1 = _1pop['Count'].to_numpy()
x1 = _1pop['Year'].to_numpy().reshape(-1, 1)

y2 = _2pop['Count'].to_numpy()
x2 = _2pop['Year'].to_numpy().reshape(-1, 1)

y3 = _3pop['Count'].to_numpy()
x3 = _3pop['Year'].to_numpy().reshape(-1, 1)


model1 = LinearRegression()
model1.fit(x1,y1)

model2 = LinearRegression()
model2.fit(x2,y2)

model3 = LinearRegression()
model3.fit(x3,y3)



l = np.arange(1800, 2200, 10).reshape(-1,1)
future1  = model1.predict(l)
future2  = model2.predict(l)
future3  = model3.predict(l)

plt.scatter(l,future1, color="green", label="China")
plt.scatter(l,future2, color="red", label="United States")
plt.scatter(l,future3, color="blue", label="India")

plt.legend()



plt.show()
