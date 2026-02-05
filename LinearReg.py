#importing libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
#import data
df = pd.read_csv("Day2-3/LinearRegressionModel/Rounded_Student_Hours_Studied_vs_Marks_Dataset.csv")

#Slicing input and output
X = df.iloc[:, : -1].values
y = df.iloc[:, -1].values
#%20 test, %80 train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
#create model
model = LinearRegression()
#model training
model.fit(X_train, y_train)
#make a prediction
prediction = model.predict(X_test)

mae = mean_absolute_error(y_test, prediction)
print("MAE: ", mae)
mse = mean_squared_error(y_test, prediction)
print("MSE: ", mse)
print("Predictions:", prediction)
print("Real values:", y_test)

plt.scatter(X, y, color="blue")
plt.plot(X, model.predict(X), color="red")
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.show()
