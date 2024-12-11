import numpy as np
import pandas as pd
import argparse
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(
    description="fitting linear regression model"
)

parser.add_argument(
    "--datapath",
    help="training and validation data path",
    default="filtered_data.csv",
)
parser.add_argument(
    "--num_epochs", 
    help="number of training epoch", 
    default=100000, 
    type=int
)

parser.add_argument(
    "--lr", 
    help="learning rate", 
    default=0.1, 
    type=float
)

args = parser.parse_args()
df = pd.read_csv(args.datapath)
X = np.array(df.iloc[:,0])
y = np.array(df.iloc[:,1])

epsilon = 1e-5
lr = args.lr
train_costs = []
val_costs = []
X = np.c_[np.ones(X.shape[0]), X]
theta = np.zeros(X.shape[1])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
m= len(y_train)

for i in range(args.num_epochs):
    pred_val = np.dot(X_train,theta)
    gradient = (1/m)*np.dot(X_train.T,pred_val-y_train)
    theta = theta - lr * gradient
    cost = (1 / (2 * m)) * np.sum((pred_val-y_train) ** 2)
    train_costs.append(cost)
    val_cost = (1 / (2 * len(y_val))) * np.sum((np.dot(X_val,theta)-y_val) ** 2)
    val_costs.append(val_cost)
    if i>0 and abs(train_costs[-1] - train_costs[-2]) < epsilon:
        #print(i+1)
        break

print(theta)
# plt.plot(train_costs, label='Training set loss')
# plt.xlabel('Iterations')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# plt.plot(val_costs, label='Validation set loss',color='orange')
# plt.xlabel('Iterations')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

