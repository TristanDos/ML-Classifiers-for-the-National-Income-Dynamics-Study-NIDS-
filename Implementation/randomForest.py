import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, ParameterGrid,
                                     cross_val_score, train_test_split)
from tqdm import tqdm

import plotter

RANDOM_STATE = 42

class RandomForestModel:
    def __init__(self, df : pd.DataFrame, target, model_path="models/RF_model.pkl", params_path="models/RF_params.pkl", metrics_path="metrics/RF_model.pkl", results_path="RF_results.txt"):
        self.model = None
        self.df = df
        self.target = target
        self.model_path = model_path
        self.metric_path = metrics_path
        self.results_path = results_path
        self.params_path = params_path
        self.params = None

        if (os.path.exists(self.params_path)):
            self.params = plotter.load_model(self.params_path)

        if (os.path.exists(self.model_path)):
            self.load_model()

            print("LOADING COMPLETED: ", self.model.get_params())

    def load_model(self, path=""):
        """Updates self.model with loaded model.
        """   
        if path == "":     
            self.model = plotter.load_model(self.model_path)
            self.params = plotter.load_model(self.params_path)
        else:
            self.model = plotter.load_model(path)
            self.params = plotter.load_model(self.params_path)

    def save_model(self, path=""):
        """Saves self.model to self.model_path
        """        
        if path == "":
            plotter.save_model(self.model, self.model_path)
            plotter.save_model(self.params, self.params_path)
        else:
            plotter.save_model(self.model, path)
            plotter.save_model(self.params, self.params_path)

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

    def train_random_forest(self, X_train, y_train):
        # Initialize and train the Random Forest model
        self.model = RandomForestClassifier(random_state=RANDOM_STATE, verbose=True)
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X, y_true, set_name="Validation", path=""):
        # Predict using the trained model (class labels)
        y_pred = self.model.predict(X)
        
        # Get predicted probabilities for the positive class (class 1)
        y_prob = self.model.predict_proba(X)[:, 1]
        
        # Evaluate metrics
        conf_matrix = confusion_matrix(y_true, y_pred)
        class_report = classification_report(y_true, y_pred)
        
        # Additional metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        auc_roc = roc_auc_score(y_true, y_prob)  # Use probabilities for AUC-ROC

        scores = {}
        scores['accuracy'] = accuracy
        scores['f1'] = f1
        scores['precision'] = precision
        scores['recall'] = recall
        scores['auc_roc'] = auc_roc

        metrics = (conf_matrix, class_report, scores)

        if path == "": 
            print("SAVING EVALUATION TO: ", self.metric_path)
            plotter.save_model(metrics, self.metric_path)
        else:
            print("SAVING EVALUATION TO: ", path)
            plotter.save_model(metrics, path)
        
        # Print results
        out = ""
        out += f"{set_name} Confusion Matrix:\n"
        out += str(conf_matrix) + "\n"
        out += f"{set_name} Classification Report:\n"
        out += str(class_report) + "\n"
        out += f"{set_name} Accuracy: {accuracy:.2f}\n"
        out += f"{set_name} F1 Score: {f1:.2f}\n"
        out += f"{set_name} Precision: {precision:.2f}\n"
        out += f"{set_name} Recall: {recall:.2f}\n"
        out += f"{set_name} AUC-ROC: {auc_roc:.2f}\n"
        
        print(out)
        return out


    def perform_grid_search(self, X_train, y_train, param_grid):
        # Convert param_grid to a list of combinations
        param_combinations = list(ParameterGrid(param_grid))

        # Initialize progress bar with total number of combinations
        with tqdm(total=len(param_combinations)) as pbar:
            best_score = -np.inf
            best_params = None
            best_model = None

            # Iterate through all combinations of hyperparameters
            for params in param_combinations:
                # Initialize model with current params
                model = RandomForestClassifier(random_state=RANDOM_STATE, **params)
                
                # Perform cross-validation
                scores = cross_val_score(model, X_train, y_train, cv=5, verbose=1, scoring="accuracy")
                
                # Calculate mean cross-validation score
                mean_score = np.mean(scores)
                
                # Check if this is the best score
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params
                    best_model = model
                
                # Update progress bar with current parameters being tested
                pbar.set_description(f"Testing: {params}")
                pbar.update(1)

        # Output best parameters and score
        print("Best Parameters:", best_params)
        print("Best Score:", best_score)

        # Initialize and train the Random Forest model
        print("RETRAINING MODEL WITH BEST PARAMS")
        model = RandomForestClassifier(random_state=RANDOM_STATE, verbose=True, **best_params)
        model.fit(X_train, y_train)

        return model, best_params
    
    def get_params(self):
        print("Model Parameters: ", self.model.get_params())
        return self.model.get_params()
    
    def optimize_params(self, param_grid, path=""):
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_data()
        self.model, self.params = self.perform_grid_search(X_train, y_train, param_grid)
        self.get_params()

        print("SAVING MODEL TO: ", self.model_path)
        self.save_model(path=self.model_path)

    def run(self, optimize=False, param_grid=None):
        # Load and split the data
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_data()

        if optimize == True:
            self.optimize_params(param_grid)

        if self.model == None and optimize == False:
            print("TRAINING MODEL: ")
            self.train_random_forest(X_train, y_train)

            print("SAVING MODEL TO: ", self.model_path)
            self.save_model(self.model_path)

        # Evaluate the model on the validation set
        validation_results = self.evaluate_model(X_val, y_val, set_name="Validation")
        
        # Evaluate the model on the test set
        testing_results = self.evaluate_model(X_test, y_test, set_name="Test")

        f = open(self.results_path, "w")
        f.write(validation_results + testing_results)
        f.close()

# Main block to execute the class methods
if __name__ == "__main__":
    OPTIMIZE = False

    # combined_df : pd.DataFrame = pd.read_pickle("CSV/waves_combined_sampled.pkl")
    combined_df : pd.DataFrame = pd.read_pickle("CSV/waves_combined_no_sampling.pkl")

    model_path = "models/RF_model.pkl"
    params_path = "models/RF_params.pkl"
    metrics_path = "metrics/RF_metrics.pkl"
    results_path = "results/RF_results.txt"

    RF1 = RandomForestModel(combined_df, combined_df['depressed'], model_path, params_path, metrics_path, results_path)

    RF1.run()

    param_grid = {
                'n_estimators': [100, 300, 500],
                'max_depth': [None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy', 'log_loss'],
                'oob_score': [True, False] 
            }

    # param_grid = {
    #             'n_estimators': [100],
    #             'max_depth': [10],
    #             'min_samples_split': [2],
    #             'min_samples_leaf': [1],
    #             'criterion': ['gini'] 
    #         }

    if OPTIMIZE:
        model_path = "models/RF_optimized_model.pkl"
        params_path = "models/RF_optimized_params.pkl"
        metrics_path = "metrics/RF_optimized_metrics.pkl"
        results_path = "results/RF_optimized_results.txt"
        
        RF2 = RandomForestModel(combined_df, combined_df['depressed'], model_path, params_path, metrics_path, results_path)
        RF2.run(optimize=True, param_grid=param_grid)
        RF2.get_params()
