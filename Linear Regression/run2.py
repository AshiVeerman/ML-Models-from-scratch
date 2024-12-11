import numpy as np
import pandas as pd
import argparse
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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
    "--batch_size", 
    help="batch size of stochastic gradient descent", 
    default=100, 
    type=int
)

parser.add_argument(
    "--lr", 
    help="learning rate", 
    default=0.01, 
    type=float
)


args = parser.parse_args()
df = pd.read_csv(args.datapath)
X = np.array(df.iloc[:,0])
y = np.array(df.iloc[:,1])
epsilon = 1e-5
lr = args.lr
batch_size = args.batch_size
costs =[]
val_costs = []
X = np.c_[np.ones(X.shape[0]), X]
theta = np.zeros(X.shape[1])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
m= len(y_train)

start_time = time.time()
avg_costs = []
for i in range(args.num_epochs):
    lr = lr * 0.99
    cnt = 0
    avg_cost=0
    for j in range (0,m, batch_size):
        cnt+=1
        X_batch = X[j:j+batch_size]
        y_batch = y[j:j+batch_size]
        pred_val_batch = np.dot(X_batch,theta)
        gradient = (1 / len(y_batch))*np.dot(X_batch.T,pred_val_batch-y_batch)
        theta = theta - lr * gradient
        cost = (1 / (2 * len(y_batch))) * np.sum((pred_val_batch-y_batch) ** 2)
        costs.append(cost)
        val_cost = (1 / (2 * len(y_val))) * np.sum((np.dot(X_val,theta)-y_val) ** 2)
        val_costs.append(val_cost)
        avg_cost+=cost
    avg_cost/=cnt
    avg_costs.append(avg_cost)
    k = 5
    moving_avg=0
    if (i>k):
        for j in range(1,k+1):
            moving_avg += abs(avg_costs[-j]-avg_costs[-j-1])
        moving_avg = moving_avg/k
        if i>k and moving_avg < epsilon:
            #print(i+1)
            break
end_time = time.time()

#print(end_time-start_time)
print(theta)
# plt.plot(costs, label='Training set loss')
# plt.plot(val_costs, label='Validation set loss')
# plt.xlabel('Iterations in one epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

