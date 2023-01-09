import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('Advertising.csv')



plt.scatter(df['TV'], df['sales'], color="red")
plt.scatter(df['radio'], df['sales'], color="blue")
plt.scatter(df['newspaper'], df['sales'], color="green")




x = df.iloc[:,1:4].to_numpy()
y = df['sales'].to_numpy()



print(len(y))

model = LinearRegression()
model.fit(x,y)
y_pred  = model.predict(x)



plt.scatter(np.array(range(0,200)).reshape(-1,1), y_pred, color="yellow")


print("error", mean_squared_error(y, y_pred))
print("accuracy", r2_score(y, y_pred))

plt.legend()
plt.show()
