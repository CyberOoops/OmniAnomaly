import numpy as np
import pandas as pd
import pickle as pkl
import sys

## preprocess for SWaT. SWaT.A2_Dec2015, version 0
df = pd.read_csv('../datasets/SWAT/SWaT_Dataset_Attack_v0.csv')
y = df['Normal/Attack'].to_numpy()
labels = []
for i in y:
    if i == 'Attack':
        labels.append(1)
    else:
        labels.append(0)
labels = np.array(labels)
assert len(labels) == 449919
pkl.dump(labels, open('processed/SWaT_test_label.pkl', 'wb'))
print('SWaT_test_label saved')

df = df.drop(columns=[' Timestamp', 'Normal/Attack'])
test = df.to_numpy(dtype=np.float32)
assert test.shape == (449919, 51)
pkl.dump(test, open('processed/SWaT_test.pkl', 'wb'))
print('SWaT_test saved')

df = pd.read_csv('../datasets/SWAT/SWaT_Dataset_Normal_v1.csv')
df = df.drop(columns=['Unnamed: 0','Unnamed: 52'])
train = df[1:].to_numpy(dtype=np.float32)
print(train.shape)
# assert train.shape == (496800, 51)
pkl.dump(train, open('processed/SWaT_train.pkl', 'wb'))
print('SWaT_train saved')

# preprocess for WADI. WADI.A1
a = str(open('../datasets/WADIA2/WADI_14days_new.csv', 'rb').read(), encoding='utf8').split('\n')[5: -1]
a = '\n'.join(a)
with open('train1.csv', 'wb') as f:
    f.write(a.encode('utf8'))
a = pd.read_csv('train1.csv', header=None)


a = a.to_numpy()[:, 3:]
nan_cols = []
for j in range(a.shape[1]):
    for i in range(a.shape[0]):
        if a[i][j] != a[i][j]:
            nan_cols.append(j)
            break
# len(nan_cols) == 9
train = np.delete(a, nan_cols, axis=1)
train=train.astype(np.float32)
print("wadi train shape ", train.shape)
# assert train.shape == (1209601, 118)
pkl.dump(train, open('processed/WADI_train.pkl', 'wb'))
print('WADI_train saved')

df = pd.read_csv('../datasets/WADIA2/WADI_attackdataLABLE.csv')
test = df.to_numpy()[2:, 3:-1]
test=test.astype(np.float32)

test = np.delete(test, nan_cols, axis=1)
print("wadi test shape ", test.shape)
# assert test.shape == (172801, 118)
pkl.dump(test, open('processed/WADI_test.pkl', 'wb'))
print('WADI_test saved')

test_label = df.to_numpy()[2:, -1]
test_label=test_label.astype(np.int32)
print(test_label.shape)
for i in range(len(test_label)):
    if test_label[i] <= 0:
        test_label[i] = 0
pkl.dump(test_label, open('processed/WADI_test_label.pkl', 'wb'))
print('WADI_test_label saved')

# WADI labels.pkl are created manually via the description file of the dataset
