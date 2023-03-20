import pandas as pd
import numpy as np
import sys

# read in data with pandas
dataframe = pd.read_csv("./data/diabetes.csv")
# dataframe = pd.read_csv("diabetes.csv")

# process data
columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in columns:
    dataframe[col].replace(0, np.NaN, inplace=True)
dataframe.dropna(inplace=True)
dataarr = dataframe.to_numpy(copy=True)

X = dataarr[:,0:8]
y = dataarr[:, 8].astype(int)

# read in input arguments
K = int(sys.argv[1])
D = int(sys.argv[2])
N = int(sys.argv[3])

# split data into train and test sets
x_train, x_test =  X[N:], X[:N]
y_train, y_test =  y[N:], y[:N]

# Step 1: standardization
mean = np.mean(x_train, axis=0)
std = np.std(x_train, axis=0)
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# Step 2: PCA
cov = np.cov(x_train, rowvar=False)
eigenvals, eigenvecs = np.linalg.eig(cov)
eigenvecs = eigenvecs[:, :D]
x_train = np.dot(x_train, eigenvecs)
x_test = np.dot(x_test, eigenvecs)

# Step 3: KNN
def knn(x_train, y_train, x_test, K):
    dists = np.linalg.norm(x_train - x_test, axis=1)
    weights = 1 / dists
    sorted_idx = np.argsort(dists)
    k_neighbors = y_train[sorted_idx[:K]]
    return np.bincount(k_neighbors, weights=weights[:K]).argmax()

y_pred = np.zeros(N)
for i in range(N):
    y_pred[i] = knn(x_train, y_train, x_test[i], K)

# output results to file
with open('knn_results.txt', 'w') as f:
    for i in range(N):
        f.write(str(int(y_pred[i])) + ' ' + str(int(y_test[i])) + '\n')

# compare with sklearn implementation
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

scaler = StandardScaler()
x_train_sklearn = scaler.fit_transform(x_train)
x_test_sklearn = scaler.transform(x_test)

pca = PCA(n_components=D, svd_solver="full")
x_train_sklearn = pca.fit_transform(x_train_sklearn)
x_test_sklearn = pca.transform(x_test_sklearn)

clf_sklearn = KNeighborsClassifier(n_neighbors=K, weights='distance', metric='euclidean')
clf_sklearn.fit(x_train_sklearn, y_train)
y_pred_sklearn = clf_sklearn.predict(x_test_sklearn)

# output results to file
with open('knn_results_sklearn.txt', 'w') as f:
    for i in range(N):
        f.write(str(y_pred_sklearn[i]) + ' ' + str(y_test[i]) + '\n')
