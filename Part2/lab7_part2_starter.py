import pandas as pd
import numpy as np

# read in data with pandas
dataframe = pd.read_csv("./data/diabetes.csv")

# process data
columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in columns:
    dataframe[col].replace(0, np.NaN, inplace=True)  # dataframe[col] is Series corresponding to col
dataframe.dropna(inplace=True) # rows with any NA will be dropped
dataarr = dataframe.to_numpy(copy=True) # convert to np array

X = dataarr[:,0:8]
y = dataarr[:, 8].astype(int)

num_classes = np.max(y) + 1

N = 20 # you need to read N as input arg
x_train, x_test =  X[N:], X[:N]
y_train, y_test =  y[N:], y[:N],

print(X.shape, y.shape)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)



# Step 1 : standarization

# Step 2 : PCA kernel learning, transformation, and dimension reduction

# Step 3: KNN
