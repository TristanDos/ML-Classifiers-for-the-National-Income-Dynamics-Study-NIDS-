import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv('wave1_select_labelled.csv')

# First, separate the target variable (label) from the features
# Assuming you have a target column in your original dataframe called 'depression_label' that indicates depression status
# If not, make sure to create that before splitting
y = df['depressed']  # Replace with your actual target column name
X = df.drop(columns=['depressed', 'w1_a_outcome', 'pid'])  # The features you've created

# Now split the data into train+validation and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Now split the train+validation set into actual train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

print(len(X_train))

# This gives you 60% train, 20% validation, and 20% test splits

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize the Logistic Regression model
log_reg = LogisticRegression(random_state=42, max_iter=1000)

# Train the model on the training data
log_reg.fit(X_train, y_train)

# Predict on the validation set
y_val_pred = log_reg.predict(X_val)

# Predict on the test set
y_test_pred = log_reg.predict(X_test)

# Accuracy
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.2f}")

# Confusion Matrix
val_conf_matrix = confusion_matrix(y_val, y_val_pred)
print("Validation Confusion Matrix:")
print(val_conf_matrix)

# Classification Report
val_class_report = classification_report(y_val, y_val_pred)
print("Validation Classification Report:")
print(val_class_report)

# Accuracy
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Confusion Matrix
test_conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Test Confusion Matrix:")
print(test_conf_matrix)

# Classification Report
test_class_report = classification_report(y_test, y_test_pred)
print("Test Classification Report:")
print(test_class_report)

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
 
# Create a pipeline with scaler and logistic regression
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, solver='saga', tol=0.1))
 
# Create a parameter grid
param_grid = {
    'logisticregression__C': [0.1, 1, 10, 100],
    'logisticregression__penalty': ['l1', 'l2']
}
 
# Create GridSearchCV object
grid_search = GridSearchCV(pipe, param_grid, cv=5)
 
# Fit the model
grid_search.fit(X_train, y_train)
 
# Print best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)
