import pandas as pd
from sklearn import preprocessing, linear_model
import numpy as np
import sklearn


data = pd.read_csv('houses_to_rent.csv', sep=',')

### LOAD THE DATA ###
print("-" * 30)
print("IMPORTING DATA ")
print("-" * 30)
data = data[['city', 'rooms', 'bathroom', 'parking spaces', 'fire insurance', 'furniture', 'rent amount', 'animal']]

### PROCESS DATA ###
data['rent amount'] = data['rent amount'].map(lambda i: int(i[2:].replace(',', '')))
data['fire insurance'] = data['fire insurance'].map(lambda i: int(i[2:].replace(',', '')))

le = preprocessing.LabelEncoder()
data['furniture'] = le.fit_transform(data['furniture'])
le1 = preprocessing.LabelEncoder()
data['animal'] = le.fit_transform(data['animal'])

print(data.head())

print("-" * 30)
print("CHECKING NULL DATA ")
print("-" * 30)
data.isnull().sum()
print(data.isnull().sum())
# data.dropna()

print("-" * 30)
print(" HEAD ")
print("-" * 30)

print(data.head())
### SPLIT DATA
print("-" * 30)
print(" SPLIT DATA ")
print("-" * 30)

x = np.array(data.drop(['rent amount'], axis=1))
y = np.array(data['rent amount'])
print('X', x.shape)
print('Y', y.shape)

xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

print('xTrain', xTrain.shape)
print('xTrain', xTest.shape)

### TRAINING ###
print("-" * 30)
print(" TRAINING ")
print("-" * 30)

model = linear_model.LinearRegression()
model.fit(xTrain, yTrain)

accuracy = model.score(xTest, yTest)
print('COEFFICIENTS', model.coef_)
print('INTERCEPT ', model.intercept_)
print('ACCURACY ', round(accuracy * 100, 3), '%')

### EVALUATION ###
print("-" * 30)
print(" MANUAL TESTING ")
print("-" * 30)

testVals = model.predict(xTest)
print(testVals.shape)

# error =[]
# for i, testVals in enumerate(testVals):
##  print(f'Actual value:{yTest[i]} Prediction Value: {int(testVals)} Error: {error[i]}')