#Importing Library
import numpy as np 
import matplotlib.pyplot as plt
from pyod.models.knn import KNN
from pyod.utils.data import generate_data, get_outliers_inliers


#Generate Data
X_train, X_test, y_train, y_test = generate_data(n_train=500, n_test=100, n_features=2,
                                                  contamination=0.1, random_state=22)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

#Train KNN Model
knn = KNN().fit(X_train)

#Predict the Outliers in the test set
y_pred_test = knn.predict(X_test)
print(y_pred_test.shape)
print(y_test.shape)

# Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_test, marker='o', label='Data Points')
plt.title('Outlier Detection using KNN')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid()
plt.show()