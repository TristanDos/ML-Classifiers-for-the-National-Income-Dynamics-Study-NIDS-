import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

'''
STRUCTURE AS FOLLOWS:

(self, neurons=64, layers=2, dropout_rate=0.2, learning_rate=0.001)
relu activation functions in hidden layers
sigmoid activation in output layer
binary cross-entropy loss function
'''

class DeepNeuralNetworkModel:
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

    def create_model(self, neurons=128, layers=5, dropout_rate=0.2, learning_rate=0.001):
        model = Sequential()
        model.add(Dense(neurons, activation='relu', input_shape=(self.X_train.shape[1],)))
        model.add(Dropout(dropout_rate))
        
        for _ in range(layers - 1):
            model.add(Dense(neurons, activation='relu'))
            model.add(Dropout(dropout_rate))
        
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def train_model(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train  # Store for use in create_model
        self.model = self.create_model()
        history = self.model.fit(X_train, y_train, 
                                 validation_data=(X_val, y_val),
                                 epochs=100, 
                                 batch_size=32, 
                                 verbose=True)
        return history

    def evaluate_model(self, X, y_true, set_name="Validation"):
        # Predict using the trained model
        y_pred = (self.model.predict(X) > 0.5).astype("int32")
        
        # Evaluate accuracy, confusion matrix, and classification report
        accuracy = accuracy_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)
        class_report = classification_report(y_true, y_pred)
        
        # Print results
        out = ""
        out += f"{set_name} Accuracy: {accuracy:.2f}" + "\n"
        out += f"{set_name} Confusion Matrix:" + "\n"
        out += str(conf_matrix) + "\n"
        out += f"{set_name} Classification Report:" + "\n"
        out += str(class_report) + "\n"
        print(out)
        return out

    def perform_grid_search(self, X_train, y_train):
        def create_model(neurons=64, layers=2, dropout_rate=0.2, learning_rate=0.001):
            model = Sequential()
            model.add(Dense(neurons, activation='relu', input_shape=(X_train.shape[1],)))
            model.add(Dropout(dropout_rate))
            
            for _ in range(layers - 1):
                model.add(Dense(neurons, activation='relu'))
                model.add(Dropout(dropout_rate))
            
            model.add(Dense(1, activation='sigmoid'))
            
            model.compile(optimizer=Adam(learning_rate=learning_rate),
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
            return model

        model = KerasClassifier(build_fn=create_model, verbose=0)

        param_grid = {
            'neurons': [32, 64, 128],
            'layers': [2, 3, 4],
            'dropout_rate': [0.1, 0.2, 0.3],
            'learning_rate': [0.001, 0.01],
            'batch_size': [32, 64],
            'epochs': [50, 100]
        }

        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=1, n_jobs=-1)
        grid_result = grid.fit(X_train, y_train)

        print("Best Parameters:", grid_result.best_params_)
        print("Best Score:", grid_result.best_score_)

        # Train the model with the best parameters
        self.model = self.create_model(**grid_result.best_params_)
        self.model.fit(X_train, y_train, epochs=grid_result.best_params_['epochs'], 
                       batch_size=grid_result.best_params_['batch_size'], verbose=0)

    def run(self):
        # Load and split the data
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_data()
        
        # Train the initial model
        self.train_model(X_train, y_train, X_val, y_val)

        # Evaluate the model on the validation set
        validation_results = self.evaluate_model(X_val, y_val, set_name="Validation")
        
        # Evaluate the model on the test set
        testing_results = self.evaluate_model(X_test, y_test, set_name="Test")
        
        # # Perform grid search to find the best hyperparameters
        # self.perform_grid_search(X_train, y_train)

        # # Re-evaluate the model with best parameters on the validation set
        # self.evaluate_model(X_val, y_val, set_name="Validation (Best Parameters)")
        
        # # Re-evaluate the model with best parameters on the test set
        # self.evaluate_model(X_test, y_test, set_name="Test (Best Parameters)")

        f = open("results_NN.txt", "w")
        f.write(validation_results + testing_results)
        f.close()
        


# Main block to execute the class methods
if __name__ == "__main__":
    # Load your data
    df = pd.read_csv('your_data.csv')  # Replace with your actual data loading method
    target = df['depressed']  # Assuming 'depressed' is your target column
    
    # Instantiate the model
    dnn_model = DeepNeuralNetworkModel(df, target)
    
    # Run the training, evaluation, and grid search
    dnn_model.run()