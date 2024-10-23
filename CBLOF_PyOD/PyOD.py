import numpy as np 
from pyod.models.cblof import CBLOF
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# Generate Sample data
X_train, X_test, y_train, y_test = generate_data(n_train=800, n_test=150, n_features=2,
                                                 contamination=0.1, random_state=22 )
# Spliting data into training and validation sets
X_train, X_val = train_test_split(X_train, test_size=0.1, random_state=22)

# Initialize the CBLOF model (Clustering-Based Local Outlier Factor)
cblof = CBLOF(contamination=0.1, random_state=22)

# Fit the Model
cblof.fit(X_train) 

# Get the prediction labels and scores for the test data
y_test_pred = cblof.predict(X_test) #binary labels (0: normal, 1: outlier)
y_test_score = cblof.decision_function(X_test) # outlier scores

# Evaluate The Model
evaluate_print('CBLOF', y_test, y_test_score)
    #output results : 
    # CBLOF ROC:CBLOF ROC:0.9965, precision @ rank n:0.9333



# Visualize the test data and highlight anomalies
plt.figure(figsize=(8, 6))

# Normal points (0 label)
plt.scatter(X_test[y_test_pred == 0][:, 0], X_test[y_test_pred == 0][:, 1], 
            color='blue', label='Normal')

# Anomaly points (1 label)
plt.scatter(X_test[y_test_pred == 1][:, 0], X_test[y_test_pred == 1][:, 1], 
            color='red', label='Anomalies')

plt.title('CBLOF Anomaly Detection')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid()
plt.show()