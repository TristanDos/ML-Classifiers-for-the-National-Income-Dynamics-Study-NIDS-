import os
import pickle

import numpy as np
import pandas as pd
import plotter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 20

class LogisticRegressionModel:
    def __init__(self, df : pd.DataFrame, target):
        # self.csv_file = csv_file
        self.model = None
        self.grid_search = None
        self.df = df
        self.target = target
        self.model_path = "models/LR_model.pkl"
        self.metric_path = "metrics/LR_metrics.pkl"
        self.results_path = "results/LR_results.txt"
        self.threshold = 0.5
        self.optimized = False

    def load_model(self, path=""):
        """Updates self.model with loaded model.
        """        
        if path == "":     
            self.model = plotter.load_model(self.model_path)
        else:
            self.model = plotter.load_model(path)

    def save_model(self, path=""):
        """Saves self.model to self.model_path
        """        
        if path == "":
            plotter.save_model(self.model, self.model_path)
        else:
            plotter.save_model(self.model, path)

    def load_data(self):
        # Load data from the provided CSV file
        df = self.df
        
        # Separate the target variable from features
        y = self.target  # Assuming 'depressed' is the target column
        X = df.drop(columns=['depressed', 'pid'])  # Drop non-feature columns
        
        # Split data into train+validation and test sets (80/20)
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
        
        # Split train+validation into train and validation sets (60/20)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=RANDOM_STATE)
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_logistic_regression(self, X_train, y_train):
        # Initialize and train the Logistic Regression model
        self.model = LogisticRegression(random_state=42, solver='newton-cholesky', max_iter=10000, penalty='l2', verbose = 1)
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X, y_true, set_name="Validation"):
        # Predict probabilities using the trained model
        y_prob = self.model.predict_proba(X)[:, 1]  # Logistic regression outputs probabilities for each class
        
        # Convert probabilities to binary predictions using threshold 0.5
        y_pred = (y_prob > self.threshold).astype("int32")
        
        # Evaluate metrics
        conf_matrix = confusion_matrix(y_true, y_pred)
        class_report = classification_report(y_true, y_pred)
        
        # Additional metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted', labels=np.unique(y_pred))
        precision = precision_score(y_true, y_pred, average='weighted', labels=np.unique(y_pred))
        recall = recall_score(y_true, y_pred, average='weighted', labels=np.unique(y_pred))
        auc_roc = roc_auc_score(y_true, y_prob)  # Use probabilities for AUC-ROC

        scores = {}
        scores['accuracy'] = accuracy
        scores['f1'] = f1
        scores['precision'] = precision
        scores['recall'] = recall
        scores['auc_roc'] = auc_roc

        # Calculate ROC curve points
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        scores['fpr'] = fpr
        scores['tpr'] = tpr
        scores['thresholds'] = thresholds

        metrics = (conf_matrix, class_report, scores)

        if self.optimized: self.metric_path = "metrics/LR_optimized_metrics.pkl"
        print("SAVING EVALUATION TO: ", self.metric_path)
        plotter.save_model(metrics, self.metric_path)
        
        # Print results
        out = ""
        out += f"{set_name} Confusion Matrix:\n"
        out += str(conf_matrix) + "\n"
        out += f"{set_name} Classification Report:\n"
        out += str(class_report) + "\n"
        out += f"{set_name} Accuracy: {accuracy:.2f}\n"
        out += f"{set_name} Precision: {precision:.2f}\n"
        out += f"{set_name} Recall: {recall:.2f}\n"
        out += f"{set_name} F1 Score: {f1:.2f}\n"
        out += f"{set_name} AUC-ROC: {auc_roc:.2f}\n"
        
        print(out)
        return out


    def perform_grid_search(self, X_train, y_train, param_grid):
        # Create a pipeline with standard scaler and logistic regression
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, solver='saga', tol=0.1))
        
        # Initialize GridSearchCV
        self.grid_search = GridSearchCV(pipe, param_grid, cv=5, verbose=1)
        
        # Perform grid search
        self.grid_search.fit(X_train, y_train)
        
        # Print best parameters and best score
        out = ""
        out += "Best Parameters: " + str(self.grid_search.best_params_) + "\n"
        out += "Best Score: " + str(self.grid_search.best_score_) + "\n"
        print(out)

        model = self.grid_search

        return model
    
    def get_params(self):
        print("Model Parameters: ", self.model.get_params())
        return self.model.get_params()
    
    def optimize_params(self, param_grid, path=""):
        self.model = self.perform_grid_search(self.X_train, self.y_train, param_grid)
        self.get_params()

        self.optimized = True

        if path == "":
            print("SAVING MODEL TO: ", self.model_path)
            self.save_model()
        else:
            self.model_path = path
            print("SAVING MODEL TO: ", path)
            self.save_model(path=path)

    def run(self, threshold=-1):

        # Load and split the data
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_data()

        self.X_train = X_train
        self.y_train = y_train
        
        # Train the Logistic Regression model
        if (os.path.exists(self.model_path)):
            self.load_model()
            print("LOADED MODEL: ", self.model.get_params())
        else:
            print("TRAINING MODEL: ")
            self.train_logistic_regression(X_train, y_train)

        if threshold != -1:
            self.threshold = threshold

        # Evaluate the model on the validation set
        validation_results = self.evaluate_model(X_val, y_val, set_name="Validation")
        
        # Evaluate the model on the test set
        testing_results = self.evaluate_model(X_test, y_test, set_name="Test")

        f = open(f"results/results_LR_threshold_{threshold}.txt", "w")
        f.write(validation_results + testing_results)
        f.close()

        #  This format {'logisticregression__C': 1, 'logisticregression__penalty': 'l1'}
        print("SAVING MODEL TO: ", self.model_path)
        self.save_model()

# Main block to execute the class methods
if __name__ == "__main__":
    OPTIMIZE = False

    # combined_df : pd.DataFrame = pd.read_pickle("CSV/waves_combined_pandas_sampling.pkl")
    combined_df : pd.DataFrame = pd.read_pickle("CSV/waves_combined_sampled.pkl")
    # combined_df : pd.DataFrame = pd.read_pickle("CSV/waves_combined_no_sampling.pkl")

    LR = LogisticRegressionModel(combined_df, combined_df['depressed'])

    LR.run(0.5)

    # Parameter grid for GridSearchCV
    param_grid = {
        'logisticregression__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'logisticregression__penalty': ['l1', 'l2']
    }

    if OPTIMIZE:
        LR.optimize_params(param_grid, "models/LR_optimized_model.pkl")
        LR.run(0.5)
