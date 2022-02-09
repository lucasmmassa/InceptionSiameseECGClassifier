import numpy as np
import pandas as pd

X_train = np.load('x_train_100.npy', allow_pickle=True)
y_train = np.load('y_train_100.npy', allow_pickle=True)
X_test = np.load('x_test_100.npy', allow_pickle=True)
y_test = np.load('y_test_100.npy', allow_pickle=True)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print(X_test.dtype)

aux = []
y1 = []
for i in range(y_train.shape[0]):
    if len(y_train[i]) == 1:
        aux.append(i)
        y1.append(y_train[i][0])

y1 = np.array(y1)
x1 = X_train[aux]

aux = []
y2 = []
for i in range(y_test.shape[0]):
    if len(y_test[i]) == 1:
        aux.append(i)
        y2.append(y_test[i][0])

y2 = np.array(y2)
x2 = X_test[aux]

print(x1.shape)
print(y1.shape)
print(x2.shape)
print(y2.shape)
# N = x1.shape[0]
# data = np.vstack([x1, x2])
#
# print(data.shape)
# for i in range(data.shape[0]):
#     data[i] = (data[i] - np.min(data[i])) / (np.max(data[i]) - np.min(data[i]))
#
# x1 = data[:N, :, :]
# x2 = data[N:, :, :]
# print(x1.shape)
# print(x2.shape)

np.save('x_train.npy', x1, allow_pickle=True)
np.save('y_train.npy', y1, allow_pickle=True)
np.save('x_test.npy', x2, allow_pickle=True)
np.save('y_test.npy', y2, allow_pickle=True)