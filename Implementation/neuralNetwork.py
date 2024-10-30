import os
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

import plotter

RANDOM_STATE = 42
NEURONS = 64
LAYERS = 3

class DeepNeuralNetworkModel(nn.Module):
    def __init__(self, input_size, neurons=64, layers=3, dropout_rate=0.2):
        super(DeepNeuralNetworkModel, self).__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_size, neurons))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout_rate))

        # Hidden layers
        for _ in range(layers - 1):
            self.layers.append(nn.Linear(neurons, neurons))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))

        # Output layer
        self.output_layer = nn.Linear(neurons, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return self.sigmoid(x)


class NeuralNetworkHandler:
    def __init__(self, df: pd.DataFrame, target, model_path="models/NN_model.pth", params_path="models/NN_params.pkl", metrics_path="metrics/NN_model.pkl", results_path="results/NN_results.txt", device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.df = df
        self.target = target
        self.scaler = StandardScaler()
        self.input_size = None  # Will be set when data is loaded
        self.threshold = 0.5
        self.model_path = model_path
        self.metric_path = metrics_path
        self.results_path = results_path
        self.params_path = params_path
        self.params = None

        if (os.path.exists(self.params_path)):
            self.params = plotter.load_model(self.params_path)

        if (os.path.exists(self.model_path)):
            self.load_model()

            print("REFITTING LOADED MODEL: ")
            X_train, X_val, X_test, y_train, y_val, y_test = self.load_data()
            # self.model = self.train_model_cv(self.model, X_train, X_val, y_train, y_val, verbose=1)
            print("LOADING COMPLETED: ", self.get_params())

    def load_model(self, path=""):
        if path == "":
            path = self.model_path
        if self.model is None:

            if self.input_size is None:
                # Load data to get input size if not already set
                self.load_data()

            if self.params is None:
                self.create_model()
            else:
                self.create_model(neurons=self.params['neurons'], layers=self.params['layers'], dropout_rate=self.params['dropout_rate'])

        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)

    def save_model(self, path=""):
        if path == "":
            path = self.model_path
        torch.save(self.model.state_dict(), path)

        plotter.save_model(self.params, self.params_path)

    def load_data(self):
        y = self.target
        X = self.df.drop(columns=['depressed', 'pid'])
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=RANDOM_STATE)
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        self.input_size = X_train.shape[1]

        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def load_data_for_cv(self):
        y = self.target
        X = self.df.drop(columns=['depressed', 'pid'])
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=RANDOM_STATE)
        
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        self.input_size = X_train.shape[1]

        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        X_test = torch.FloatTensor(X_test).to(self.device)
        y_train = torch.FloatTensor(y_train.values).to(self.device)
        y_val = torch.FloatTensor(y_val.values).to(self.device)
        y_test = torch.FloatTensor(y_test.values).to(self.device)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def create_model(self, neurons=NEURONS, layers=LAYERS, dropout_rate=0.2):
        if self.input_size is None:
            raise ValueError("Input size is not set. Please load data first.")
        self.model = DeepNeuralNetworkModel(self.input_size, neurons, layers, dropout_rate).to(self.device)

    def train_model(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, learning_rate=0.001, verbose=0):
        input_size = X_train.shape[1]
        self.input_size = input_size

        self.model = DeepNeuralNetworkModel(input_size).to(self.device)

        # Define loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(self.device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).to(self.device)

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_train_tensor).squeeze()
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

            if epoch % batch_size == 0:
                # Validation phase
                self.model.eval()
                val_outputs = self.model(X_val_tensor).squeeze()
                val_loss = criterion(val_outputs, y_val_tensor).item()
                if verbose == 1: print(f'Epoch {epoch}/{epochs} - Training Loss: {loss.item():.4f} - Validation Loss: {val_loss:.4f}')

        self.optimizer = optimizer
        self.epoch = epochs

        return self.model, optimizer, epochs

    def train_model_cv(self, model, X_train, y_train, X_val, y_val, learning_rate=0.001, epochs=10000, batch_size=32, verbose=1):
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # train_dataset = TensorDataset(X_train, y_train)
        # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train).squeeze()
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            if epoch % batch_size == 0:
                # Validation phase
                model.eval()
                val_outputs = model(X_val).squeeze()
                val_loss = criterion(val_outputs, y_val).item()
                if verbose == 1: print(f'Epoch {epoch}/{epochs} - Training Loss: {loss.item():.4f} - Validation Loss: {val_loss:.4f}')

            # for batch_X, batch_y in train_loader:
            #     optimizer.zero_grad()
            #     outputs = model(batch_X)
            #     loss = criterion(outputs, batch_y.unsqueeze(1))
            #     loss.backward()
            #     optimizer.step()
            
            # if verbose and (epoch + 1) % 50 == 0:
            #     model.eval()
            #     with torch.no_grad():
            #         val_outputs = model(X_val)
            #         val_loss = criterion(val_outputs, y_val.unsqueeze(1))
            #     print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

        return model

    def evaluate_model_cv(self, model, X, y_true, set_name="Validation", verbose=1):
        model.eval()
        with torch.no_grad():
            y_prob = model(X).cpu().numpy()
        y_pred = (y_prob > self.threshold).astype("int32")
        y_true = y_true.cpu().numpy()

        conf_matrix = confusion_matrix(y_true, y_pred)
        class_report = classification_report(y_true, y_pred)

        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        auc_roc = roc_auc_score(y_true, y_prob)

        scores = {}
        scores['accuracy'] = accuracy
        scores['f1'] = f1
        scores['precision'] = precision
        scores['recall'] = recall
        scores['auc_roc'] = auc_roc

        metrics = (conf_matrix, class_report, scores)

        out = f"{set_name} Confusion Matrix:\n{conf_matrix}\n"
        out += f"{set_name} Classification Report:\n{class_report}\n"
        out += f"{set_name} Accuracy: {accuracy:.2f}\n"
        out += f"{set_name} F1 Score: {f1:.2f}\n"
        out += f"{set_name} Precision: {precision:.2f}\n"
        out += f"{set_name} Recall: {recall:.2f}\n"
        out += f"{set_name} AUC-ROC: {auc_roc:.2f}\n"
        if verbose == 1: print(out)

        return out, metrics

    def evaluate_model(self, X, y_true, set_name="Validation", verbose=0, path=""):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        self.model.eval()

        y_prob = self.model(X_tensor).cpu().detach().numpy().squeeze()
        y_pred = (y_prob > self.threshold).astype(int)

        conf_matrix = confusion_matrix(y_true, y_pred)
        class_report = classification_report(y_true, y_pred)

        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        auc_roc = roc_auc_score(y_true, y_prob)

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

        out = f"{set_name} Confusion Matrix:\n{conf_matrix}\n"
        out += f"{set_name} Classification Report:\n{class_report}\n"
        out += f"{set_name} Accuracy: {accuracy:.2f}\n"
        out += f"{set_name} F1 Score: {f1:.2f}\n"
        out += f"{set_name} Precision: {precision:.2f}\n"
        out += f"{set_name} Recall: {recall:.2f}\n"
        out += f"{set_name} AUC-ROC: {auc_roc:.2f}\n"
        if verbose == 1: print(out)

        return out, metrics

    def perform_grid_search_valid(self, X_train, y_train, X_val, y_val, param_grid):
        input_dim = X_train.shape[1]
        best_score = 0
        best_params = None
        best_model = None

         # Calculate the total number of combinations
        total_combinations = np.prod([len(param_grid[key]) for key in param_grid])

        # Use tqdm for progress bar
        with tqdm(total=total_combinations, desc="Tuning Hyperparameters", unit="combination") as pbar:
            # Generate all combinations of hyperparameters
            for neurons, layers, dropout_rate, learning_rate, batch_size, epochs in product(
                    param_grid['neurons'], param_grid['layers'], param_grid['dropout_rate'], 
                    param_grid['learning_rate'], param_grid['batch_size'], param_grid['epochs']):
                
                print(f"Validating: ", neurons, layers, dropout_rate, learning_rate, batch_size, epochs)
                model = DeepNeuralNetworkModel(input_dim, neurons=neurons, layers=layers, dropout_rate=dropout_rate).to(self.device)
                model, _, _ = self.train_model(X_train, y_train, X_val, y_val, 
                                        learning_rate=learning_rate, epochs=epochs, batch_size=batch_size, verbose=1)
                
                # Evaluate the model
                out, metrics = self.evaluate_model(X_val, y_val)
                conf_matrix, class_report, accuracy, f1, precision, recall, auc_roc = metrics

                score = accuracy
                
                if score > best_score:
                    best_score = score
                    best_params = {'neurons': neurons, 'layers': layers, 'dropout_rate': dropout_rate,
                                'learning_rate': learning_rate, 'batch_size': batch_size, 'epochs': epochs}
                    best_model = model
                
                # Update tqdm progress bar
                pbar.update(1)
            
        print(f"Best Parameters: {best_params}")
        print(f"Best Score: {best_score}")
        
        return best_model

    def perform_grid_search_cv(self, X_train, y_train, param_grid, n_splits=5):
        best_score = 0
        best_params = None
        best_model = None

        # Calculate the total number of combinations
        total_combinations = np.prod([len(param_grid[key]) for key in param_grid])

        # Define K-Fold cross-validation
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

        # Convert PyTorch tensors back to numpy for sklearn compatibility
        X_train_np = X_train.cpu().numpy()
        y_train_np = y_train.cpu().numpy()

        # Use tqdm for progress bar
        with tqdm(total=total_combinations, desc="Tuning Hyperparameters", unit="combination") as pbar:
            # Generate all combinations of hyperparameters
            for neurons, layers, dropout_rate, learning_rate, batch_size, epochs in product(
                    param_grid['neurons'], param_grid['layers'], param_grid['dropout_rate'], 
                    param_grid['learning_rate'], param_grid['batch_size'], param_grid['epochs']):
                
                print(f"\nValidating: neurons={neurons}, layers={layers}, dropout_rate={dropout_rate}, "
                      f"learning_rate={learning_rate}, batch_size={batch_size}, epochs={epochs}")
                
                # Perform cross-validation
                cv_scores = []
                for train_index, val_index in kf.split(X_train_np):
                    X_train_fold = torch.FloatTensor(X_train_np[train_index]).to(self.device)
                    X_val_fold = torch.FloatTensor(X_train_np[val_index]).to(self.device)
                    y_train_fold = torch.FloatTensor(y_train_np[train_index]).to(self.device)
                    y_val_fold = torch.FloatTensor(y_train_np[val_index]).to(self.device)

                    model = DeepNeuralNetworkModel(self.input_size, neurons=neurons, layers=layers, dropout_rate=dropout_rate).to(self.device)
                    model = self.train_model_cv(model, X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                                             learning_rate=learning_rate, epochs=epochs, batch_size=batch_size, verbose=0)
                    
                    # Evaluate the model
                    _, metrics = self.evaluate_model_cv(model, X_val_fold, y_val_fold, verbose=0)
                    _, _, scores = metrics
                    cv_scores.append(scores['accuracy'])

                # Calculate mean cross-validation score
                mean_cv_score = np.mean(cv_scores)
                print(f"Mean CV Score: {mean_cv_score:.4f}")

                if mean_cv_score > best_score:
                    best_score = mean_cv_score
                    best_params = {'neurons': neurons, 'layers': layers, 'dropout_rate': dropout_rate,
                                   'learning_rate': learning_rate, 'batch_size': batch_size, 'epochs': epochs}
                    best_model = model

                # Update tqdm progress bar
                pbar.update(1)

        print(f"Best Parameters: {best_params}")
        print(f"Best Score: {best_score}")

        return best_model, best_params

    def get_params(self):
        print("Model Parameters: ", self.model.named_parameters, self.model.layers)
        return self.model.named_parameters, self.model.layers

    def optimize_params(self, param_grid):
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_data_for_cv()
        self.model, self.params = self.perform_grid_search_cv(X_train, y_train, param_grid)
        self.get_params()

        print("SAVING MODEL TO: ", self.model_path)
        self.save_model(path=self.model_path)

    def run(self, threshold=-1, optimize=False, param_grid=None):
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_data()

        if threshold != -1:
            self.threshold = threshold

        if optimize == True:
            self.optimize_params(param_grid)

        if self.model == None and optimize == False:
            print("TRAINING MODEL: ")
            self.train_model(X_train, y_train, X_val, y_val, verbose=1)

            print("SAVING MODEL TO: ", self.model_path)
            self.save_model(self.model_path)

        validation_results, _ = self.evaluate_model(X_val, y_val, set_name="Validation", verbose=1)
        testing_results, _ = self.evaluate_model(X_test, y_test, set_name="Test", verbose=1)

        f = open(self.results_path, "w")
        f.write(validation_results + testing_results)
        f.close()

# Main block to execute the class methods
if __name__ == "__main__":
    OPTIMIZE = False

    # combined_df : pd.DataFrame = pd.read_pickle("CSV/waves_combined_sampled.pkl")
    combined_df : pd.DataFrame = pd.read_pickle("CSV/waves_combined_no_sampling.pkl")

    model_path = "models/NN_model.pth"
    params_path = "models/NN_params.pkl"
    metrics_path = "metrics/NN_metrics.pkl"
    results_path = "results/NN_results.txt"

    NN1 = NeuralNetworkHandler(combined_df, combined_df['depressed'], model_path, params_path, metrics_path, results_path)

    NN1.run(threshold=0.5)

    param_grid = {
        'neurons': [32, 64, 128],
        'layers': [3, 4, 5],
        'dropout_rate': [0.1, 0.2, 0.3],
        'learning_rate': [0.001, 0.01],
        'batch_size': [32, 64],
        'epochs': [100000]
    }

    # param_grid = {
    #     'neurons': [32],
    #     'layers': [3],
    #     'dropout_rate': [0.1],
    #     'learning_rate': [0.01],
    #     'batch_size': [32],
    #     'epochs': [10]
    # }

    if OPTIMIZE:
        model_path = "models/NN_optimized_model.pth"
        params_path = "models/NN_optimized_params.pkl"
        metrics_path = "metrics/NN_optimized_metrics.pkl"
        results_path = "results/NN_optimized_results.txt"

        NN2 = NeuralNetworkHandler(combined_df, combined_df['depressed'], model_path, params_path, metrics_path, results_path)
        NN2.run(threshold=0.5, optimize=True, param_grid=param_grid)
        NN2.get_params()