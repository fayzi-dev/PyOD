# Import necessary Libraries
import numpy as np
from pyod.models.ecod import ECOD
from pyod.utils.data import generate_data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


# Generate synthetic data with outliers
# n_train: number of training points
# n_test: number of testing points
# n_features: number of features (dimensions)
# contamination: percentage of outliers (anomalies) in the data
X_train, X_test, y_train, y_test = generate_data(n_train=200, n_test=50,
                                                  n_features=2, contamination=0.1)

# Initialize the ECOD model
model = ECOD()

# Fit the model on the training data
model.fit(X_train)

# Predict the anomaly scores for the test data
anomaly_scores = model.decision_function(X_test)

predictions = model.predict(X_test)
# Print Results 
print('Anomaly Scores:', anomaly_scores)

print('Predictions (0: inlier, 1: outlier) :', predictions )

print('True Labels (0: inlier, 1: outlier) :', y_test)

# Plot the data 
plt.figure(figsize=(10,10))

# plot normal points
sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=predictions, 
                palette='coolwarm', size=60, style=predictions)

# Add labels and title to the plot
plt.title("ECOD Outlier Detection")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend(["Inliers", "Outliers"], loc='upper left')
plt.show()
