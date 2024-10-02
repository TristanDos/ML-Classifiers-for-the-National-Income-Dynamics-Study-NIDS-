import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, ParameterGrid
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

class SVMModel:
    def __init__(self, df: pd.DataFrame, target):
        self.model = None
        self.df = df
        self.target = target
        self.scaler = StandardScaler()

    def load_data(self):
        # Separate the target variable from features
        y = self.target
        X = self.df.drop(columns=['depressed', 'pid'])  # Drop non-feature columns
        
        # Split data into train+validation and test sets (80/20)
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Split train+validation into train and validation sets (60/20)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)
        
        # Scale the features
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_svm(self, X_train, y_train):
        # Initialize and train the SVM model
        self.model = SVC(random_state=42)
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
        # Define the parameter grid
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf', 'linear', 'poly'],
            'gamma': ['scale', 'auto', 0.1, 1],
            'degree': [2, 3, 4]  # Only used by poly kernel
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
                model = SVC(random_state=42, **params)
                
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

        # Train the model with the best parameters
        self.model = SVC(random_state=42, **best_params)
        self.model.fit(X_train, y_train)

    def run(self):
        # Load and split the data
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_data()
        
        # Train the SVM model
        self.train_svm(X_train, y_train)

        # Evaluate the model on the validation set
        self.evaluate_model(X_val, y_val, set_name="Validation")
        
        # Evaluate the model on the test set
        self.evaluate_model(X_test, y_test, set_name="Test")
        
        # Perform grid search to find the best hyperparameters
        self.perform_grid_search(X_train, y_train)

        # Re-evaluate the model with best parameters on the validation set
        self.evaluate_model(X_val, y_val, set_name="Validation (Best Parameters)")
        
        # Re-evaluate the model with best parameters on the test set
        self.evaluate_model(X_test, y_test, set_name="Test (Best Parameters)")


# Main block to execute the class methods
if __name__ == "__main__":
    # Load your data
    df = pd.read_csv('your_data.csv')  # Replace with your actual data loading method
    target = df['depressed']  # Assuming 'depressed' is your target column
    
    # Instantiate the model
    svm_model = SVMModel(df, target)
    
    # Run the training, evaluation, and grid search
    svm_model.run()