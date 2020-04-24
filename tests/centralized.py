from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
import constants as consts
seed = consts.SEED
points = consts.POINTS
features = consts.FEATURES
var = consts.VAR

np.random.seed(seed)

# create feature vector
x_train = np.random.normal(loc=0, scale=1, size=(points, features))
x_train[0, 0] = 1
# this w is the true coefficients
w_true = np.random.normal(loc=0, scale=1, size=features)

# noise to differentiate train data
b = np.random.normal(loc=0, scale=var * var, size=1)
y_train = np.zeros(points)
y_true = np.zeros(points)
# create target variable
for i in range(points):
    y_train[i] = np.dot(x_train[i].transpose(), w_true) + b
    y_true[i] = np.dot(x_train[i].transpose(), w_true)

# print(x_train)
# print(y_train)

# train the model with sklearn
reg = linear_model.LinearRegression()
reg.fit(x_train, y_train)

# see real_data vs predicted_data
plt.plot(y_true, color='red', label='Real data')
plt.plot(y_train, color='blue', label='Predicted data')
plt.title('Prediction')
plt.legend()
plt.show()

# scatter diagram for real vs predicted
fig, ax = plt.subplots()
ax.scatter(y_train, y_true, edgecolors=(0, 0, 0))
ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--',
        lw=2)
ax.set_xlabel('Predicted RUL', fontsize=18)
ax.set_ylabel('Actual RUL', fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_title('Centralized Predicted y vs Actual y', fontsize=20)
plt.show()
