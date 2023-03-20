import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

# read in data with pandas
dataframe = pd.read_csv("diabetes.csv")

# process data
columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in columns:
    dataframe[col].replace(0, np.NaN, inplace=True)  # dataframe[col] is Series corresponding to col
dataframe.dropna(inplace=True) # rows with any NA will be dropped
dataarr = dataframe.to_numpy(copy=True) # convert to np array

# read input arguments
K = int(sys.argv[1])
D = int(sys.argv[2])
N = int(sys.argv[3])

X = dataarr[:,0:8]
y = dataarr[:, 8].astype(int)

# split data into training and testing sets
x_train, x_test =  X[N:], X[:N]
y_train, y_test =  y[N:], y[:N],

# Step 1 : standarization
scaler = StandardScaler().fit(x_train)
x_train_std = scaler.transform(x_train)
x_test_std = scaler.transform(x_test)

# Step 2 : PCA kernel learning, transformation, and dimension reduction
pca = PCA(n_components=D, svd_solver='full').fit(x_train_std)
x_train_pca = pca.transform(x_train_std)
x_test_pca = pca.transform(x_test_std)

# Step 3: KNN
knn = KNeighborsClassifier(n_neighbors=K, metric='euclidean').fit(x_train_pca, y_train)
y_pred = knn.predict(x_test_pca)

# write results to file
with open("knn_results.txt", "w") as f:
    for i in range(N):
        f.write(str(y_pred[i]) + " " + str(y_test[i]) + "\n")

# compare with Sklearn implementation
knn_sk = KNeighborsClassifier(n_neighbors=K, metric='euclidean')
knn_sk.fit(x_train_pca, y_train)
y_pred_sk = knn_sk.predict(x_test_pca)

# write Sklearn results to file
with open("knn_results_sklearn.txt", "w") as f:
    for i in range(N):
        f.write(str(y_pred_sk[i]) + " " + str(y_test[i]) + "\n")
