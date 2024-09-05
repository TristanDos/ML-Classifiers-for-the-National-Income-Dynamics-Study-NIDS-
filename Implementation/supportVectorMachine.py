# import pandas as pd
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import make_pipeline
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# class SVMModel:
#     def __init__(self, csv_file):
#         self.csv_file = csv_file
#         self.model = None
#         self.grid_search = None

#     def load_data(self):
#         # Load data from the provided CSV file
#         df = pd.read_csv(self.csv_file)
        
#         # Separate the target variable from features
#         y = df['depressed']  # Assuming 'depressed' is the target column
#         X = df.drop(columns=['depressed', 'pid'])  # Drop non-feature columns
        
#         # Split data into train+validation and test sets (80/20)
#         X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
#         # Split train+validation into train and validation sets (60/20)
#         X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)
        
#         return X_train, X_val, X_test, y_train, y_val, y_test

#     def train_svm(self, X_train, y_train):
#         # Initialize and train the SVM model
#         self.model = SVC(random_state=42)
#         self.model.fit(X_train, y_train)

#     def evaluate_model(self, X, y_true, set_name="Validation"):
#         # Predict using the trained model
#         y_pred = self.model.predict(X)
        
#         # Evaluate accuracy, confusion matrix, and classification report
#         accuracy = accuracy_score(y_true, y_pred)
#         conf_matrix = confusion_matrix(y_true, y_pred)
#         class_report = classification_report(y_true, y_pred)
        
#         # Print results
#         print(f"{set_name} Accuracy: {accuracy:.2f}")
#         print(f"{set_name} Confusion Matrix:")
#         print(conf_matrix)
#         print(f"{set_name} Classification Report:")
#         print(class_report)

#     def perform_grid_search(self, X_train, y_train):
#         # Create a pipeline with standard scaler and SVM
#         pipe = make_pipeline(StandardScaler(), SVC())
        
#         # Parameter grid for GridSearchCV
#         param_grid = {
#             'svc__C': [0.1, 1, 10, 100],        # Regularization parameter
#             'svc__kernel': ['linear', 'rbf'],    # Kernels to try
#             'svc__gamma': ['scale', 'auto']      # Kernel coefficient
#         }
        
#         # Initialize GridSearchCV
#         self.grid_search = GridSearchCV(pipe, param_grid, cv=5)
        
#         # Perform grid search
#         self.grid_search.fit(X_train, y_train)
        
#         # Print best parameters and best score
#         print("Best Parameters:", self.grid_search.best_params_)
#         print("Best Score:", self.grid_search.best_score_)

#     def run(self):
#         # Load and split the data
#         X_train, X_val, X_test, y_train, y_val, y_test = self.load_data()
        
#         # Train the SVM model
#         self.train_svm(X_train, y_train)

#         # Evaluate the model on the validation set
#         self.evaluate_model(X_val, y_val, set_name="Validation")
        
#         # Evaluate the model on the test set
#         self.evaluate_model(X_test, y_test, set_name="Test")
        
#         # Perform grid search to find the best hyperparameters
#         self.perform_grid_search(X_train, y_train)


# # Main block to execute the class methods
# if __name__ == "__main__":
#     # Instantiate the SVM model with the CSV file path
#     svm_model = SVMModel('wave1_select_labelled.csv')
    
#     # Run the training, evaluation, and grid search
#     svm_model.run()
