import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#dane z biblioteki sclearn z uniwersytetu w winscousin (nie wymaga używania pliku csv)
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
print(breast_cancer_dataset)

# dane do bramki
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names)
data_frame.head() #wypisanie pierwszych rzędów

# dodawanie kolumny 'target'
data_frame['label'] = breast_cancer_dataset.target
data_frame.tail()

# numer kolumn i rzędów
data_frame.shape
# wziecie informacji
data_frame.info()

# spr brakujących wartości
data_frame.isnull().sum()
# statistical measures about the data
data_frame.describe()
# checking the distribution of Target Varibale
data_frame['label'].value_counts()
data_frame.groupby('label').mean()
X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2) #2-% i ziarno generatora liczb losowych
print(X.shape, X_train.shape, X_test.shape)

X_train_rf, X_test_rf, Y_train_rf, Y_test_rf = train_test_split(X, Y, test_size=0.2, random_state=2)
X_train_knn, X_test_knn, Y_train_knn, Y_test_knn = train_test_split(X, Y, test_size=0.2, random_state=2)

#plt.style.use('seaborn')
#plt.figure(figsize = (100,100))
#plt.scatter(X[:,0], X[:,1], marker= '*',s=100,edgecolors='black')
#plt.show()


#model training
model = LogisticRegression()
# trening modelu regresji
model.fit(X_train, Y_train)

#model random forest
model_rf = RandomForestClassifier()
#model random forest
model_rf.fit(X_train_rf,Y_train_rf)

#model knajblizszych sąsiadów
model_knn = KNeighborsClassifier()
#model knajblizszych sąsiadów
model_knn.fit(X_train_knn,Y_train_knn)


# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Dokładność danych treningowych dla Regresji = ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Dokładność danych testowych dla Regresji= ', test_data_accuracy)

# accuracy on training data RANODM FOREST
X_rf_train_prediction = model_rf.predict(X_train_rf)
training_data_accuracy_rf = accuracy_score(Y_train_rf, X_rf_train_prediction)
print('Dokładność danych treningowych dla Random Forest = ', training_data_accuracy_rf)

# accuracy on test data
X_rf_test_prediction = model_rf.predict(X_test_rf)
test_data_accuracy_rf = accuracy_score(Y_test_rf, X_rf_test_prediction)
print('Dokładność danych testowych dla Random Forest = ', test_data_accuracy_rf)

# accuracy on training data K NAJBLIZSZYCH SĄSIADOW YOLOOOOOO
X_knn_train_prediction = model_knn.predict(X_train_knn)
training_data_accuracy_knn = accuracy_score(Y_train_knn, X_knn_train_prediction)
print('Dokładność danych treningowych dla K-Neighbours = ', training_data_accuracy_knn)

# accuracy on test data
X_knn_test_prediction = model_knn.predict(X_test_knn)
test_data_accuracy_knn = accuracy_score(Y_test_knn, X_knn_test_prediction)
print('Dokładność danych testowych dla K-Neighbours = ', test_data_accuracy_knn)



input_data = (13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259)

# change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for one datapoint
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

prediction = model_rf.predict(input_data_reshaped)
print(prediction)

prediction = model_knn.predict(input_data_reshaped)
print(prediction)


if (prediction[0] == 0):
  print('Rak piersi jest złośliwy :,(')

else:
  print('Rak piersi jest łagodny <3')
