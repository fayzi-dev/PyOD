# Import necessary Libraries 
from pyod.models.iforest import IForest
from pyod.utils.data import generate_data , evaluate_print
from pyod.utils.example import visualize

#  Generate Sample Data
X_train, X_test, y_train, y_test = generate_data(
    n_train=1000, n_test=200, n_features=2, contamination=0.1, random_state=22
)
# Initialize the density-based Isolation Forest (DIF) Detector
model = IForest(contamination=0.1, behaviour='new', random_state=22)

#  Fit the model on training data
model.fit(X_train)

# Get the prediction labels and outlier scores 
y_train_pred = model.labels_ #Predictions on the training data
y_train_scores = model.decision_scores_ # Outlier scores on the training data
y_test_pred = model.predict(X_test) # Predictions on the test data
y_test_scores = model.decision_function(X_test) # Otlier scores on the test data

# Evaluate the model
evaluate_print('IForest on the Trainig data: ', y_train, y_train_scores)
evaluate_print('IForest on the Test data :', y_test, y_test_scores)
#  Output :
# IForest on the Trainig data:  ROC:0.9958, precision @ rank n:0.95
# IForest on the Test data : ROC:0.9958, precision @ rank n:0.9

# Visualize the results :
visualize('IForest', X_train, y_train, X_test, y_test, y_train_pred, y_test_pred)

# Confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_test_pred))
#  Output :
    # [[180   0]
    # [  3  17]]

    #     A **confusion matrix** is a table used to evaluate the performance of a classification model by comparing its predicted values with the actual values. It provides a breakdown of the correct and incorrect predictions, allowing for a more granular understanding of how well the model is performing.

    # ### Structure of a Confusion Matrix
    # For a binary classification problem, a confusion matrix typically has four values:

    # |                | Predicted Positive | Predicted Negative |
    # |----------------|--------------------|--------------------|
    # | **Actual Positive** | True Positive (TP)      | False Negative (FN)     |
    # | **Actual Negative** | False Positive (FP)     | True Negative (TN)      |
