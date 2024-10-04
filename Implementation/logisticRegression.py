import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle


class LogisticRegressionModel:
    def __init__(self, df : pd.DataFrame, target):
        # self.csv_file = csv_file
        self.model = None
        self.grid_search = None
        self.df = df
        self.target = target

    def load_data(self):
        # Load data from the provided CSV file
        df = self.df
        
        # Separate the target variable from features
        y = self.target  # Assuming 'depressed' is the target column
        X = df.drop(columns=['depressed', 'pid'])  # Drop non-feature columns
        
        # Split data into train+validation and test sets (80/20)
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Split train+validation into train and validation sets (60/20)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_logistic_regression(self, X_train, y_train):
        # Initialize and train the Logistic Regression model
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X, y_true, set_name="Validation"):
        # Predict using the trained model
        y_pred = self.model.predict(X)
        
        # Evaluate accuracy, confusion matrix, and classification report
        accuracy = accuracy_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)
        class_report = classification_report(y_true, y_pred)
        
        # Print results
        out = ""
        out += f"{set_name} Accuracy: {accuracy:.2f}" + "\n"
        out += f"{set_name} Confusion Matrix:" + "\n"
        out += conf_matrix + "\n"
        out += f"{set_name} Classification Report:" + "\n"
        out += class_report + "\n"
        print(out)
        return out

    def perform_grid_search(self, X_train, y_train):
        # Create a pipeline with standard scaler and logistic regression
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, solver='saga', tol=0.1))
        
        # Parameter grid for GridSearchCV
        param_grid = {
            'logisticregression__C': [0.1, 1, 10, 100],
            'logisticregression__penalty': ['l1', 'l2']
        }
        
        # Initialize GridSearchCV
        self.grid_search = GridSearchCV(pipe, param_grid, cv=5)
        
        # Perform grid search
        self.grid_search.fit(X_train, y_train)
        
        # Print best parameters and best score
        out = ""
        out += "Best Parameters:", self.grid_search.best_params_ + "\n"
        out += "Best Score:", self.grid_search.best_score_ + "\n"
        print(out)
        return out, self.grid_search.best_params_

    def run(self):
        # Load and split the data
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_data()
        
        # Train the Logistic Regression model
        self.train_logistic_regression(X_train, y_train)

        # Evaluate the model on the validation set
        validation_results = self.evaluate_model(X_val, y_val, set_name="Validation")
        
        # Evaluate the model on the test set
        testing_results = self.evaluate_model(X_test, y_test, set_name="Test")
        
        # Perform grid search to find the best hyperparameters
        cv_results = ""
        cv_results, params = self.perform_grid_search(X_train, y_train)

        f = open("results_LR.txt", "w")
        f.write(validation_results + testing_results + cv_results)
        f.close()

        # import pickle
        # d = { "abc" : [1, 2, 3], "qwerty" : [4,5,6] }
        # afile = open(r'C:\d.pkl', 'wb')
        # pickle.dump(d, afile)
        # afile.close()

        # #reload object from file
        # file2 = open(r'C:\d.pkl', 'rb')
        # new_d = pickle.load(file2)
        # file2.close()

        # #print dictionary object loaded from file
        # print new_d

        #  This format {'logisticregression__C': 1, 'logisticregression__penalty': 'l1'}
        f = open("params_LR.pkl", 'wb')
        pickle.dump(params, f)
        f.close()


# Main block to execute the class methods
if __name__ == "__main__":
    # Instantiate the model with the CSV file path
    lr_model = LogisticRegressionModel('wave1_select_labelled.csv')
    
    # Run the training, evaluation, and grid search
    lr_model.run()
