import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm


class RandomForestModel:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.model = None
        self.grid_search = None

    def load_data(self):
        # Load data from the provided CSV file
        df = pd.read_csv(self.csv_file)
        
        # Separate the target variable from features
        y = df['depressed']  # Assuming 'depressed' is the target column
        X = df.drop(columns=['depressed', 'pid'])  # Drop non-feature columns
        
        # Split data into train+validation and test sets (80/20)
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Split train+validation into train and validation sets (60/20)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_random_forest(self, X_train, y_train):
        # Initialize and train the Random Forest model
        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X, y_true, set_name="Validation"):
        # Predict using the trained model
        y_pred = self.model.predict(X)
        
        # Evaluate accuracy, confusion matrix, and classification report
        accuracy = accuracy_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)
        class_report = classification_report(y_true, y_pred)
        
        # Print results
        print(f"{set_name} Accuracy: {accuracy:.2f}")
        print(f"{set_name} Confusion Matrix:")
        print(conf_matrix)
        print(f"{set_name} Classification Report:")
        print(class_report)

    def perform_grid_search(self, X_train, y_train):
        # # Parameter grid for GridSearchCV
        # param_grid = {
        #     'n_estimators': [100, 200, 300],
        #     'max_depth': [10, 20, 30, None],
        #     'min_samples_split': [2, 5, 10],
        #     'min_samples_leaf': [1, 2, 4]
        # }
        
        # # Initialize GridSearchCV
        # self.grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, verbose=2)
        
        # # Perform grid search
        # self.grid_search.fit(X_train, y_train)
        
        # # Print best parameters and best score
        # print("Best Parameters:", self.grid_search.best_params_)
        # print("Best Score:", self.grid_search.best_score_)
        # Define the parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        # Convert param_grid to a list of combinations
        param_combinations = list(ParameterGrid(param_grid))

        # Initialize progress bar with total number of combinations
        with tqdm(total=len(param_combinations)) as pbar:
            best_score = -np.inf
            best_params = None

            # Iterate through all combinations of hyperparameters
            for params in param_combinations:
                # Initialize model with current params
                model = RandomForestClassifier(random_state=42, **params)
                
                # Perform cross-validation
                scores = cross_val_score(model, X_train, y_train, cv=5)
                
                # Calculate mean cross-validation score
                mean_score = np.mean(scores)
                
                # Check if this is the best score
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params
                
                # Update progress bar with current parameters being tested
                pbar.set_description(f"Testing: {params}")
                pbar.update(1)

        # Output best parameters and score
        print("Best Parameters:", best_params)
        print("Best Score:", best_score)

    def run(self):
        # Load and split the data
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_data()
        
        # Train the Random Forest model
        self.train_random_forest(X_train, y_train)

        # Evaluate the model on the validation set
        self.evaluate_model(X_val, y_val, set_name="Validation")
        
        # Evaluate the model on the test set
        self.evaluate_model(X_test, y_test, set_name="Test")
        
        # Perform grid search to find the best hyperparameters
        self.perform_grid_search(X_train, y_train)


# Main block to execute the class methods
if __name__ == "__main__":
    # Instantiate the model with the CSV file path
    rf_model = RandomForestModel('wave1_select_labelled.csv')
    
    # Run the training, evaluation, and grid search
    rf_model.run()
