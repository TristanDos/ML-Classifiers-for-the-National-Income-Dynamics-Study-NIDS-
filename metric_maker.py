import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
import pickle

class Maker():
    def __init__(self, number_test = 16313, verbose=True):
        self.accuracy = 0
        self.f1 = 0
        self.precision = 0
        self.recall = 0
        self.auc = 0
        self.np_cm = None

        # Example true labels and predictions
        self.y_true = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0]  # Actual labels
        self.y_pred = [0, 1, 0, 0, 0, 1, 1, 1, 0, 1]  # Predicted labels

        # Calculate confusion matrix
        cm = confusion_matrix(self.y_true, self.y_pred)

        # 121936
        total = int(121936 * 0.2)
        self.number_test = number_test
        
        self.verbose = verbose
        if self.verbose: print(number_test)

    def set_vals(self, diag_drift, diag_split, undiag_split):
        self.diag_drift = diag_drift
        self.diag_split = diag_split
        self.undiag_split = undiag_split
    
    def get_metrics(self):
        number_test = self.number_test

        top_l = int(number_test * diag_split)
        bottom_r = number_test - top_l - diag_drift
        bottom_l = int(diag_drift * undiag_split)
        top_r = number_test - top_l - bottom_l - bottom_r

        total = top_l + bottom_r + bottom_l + top_r

        discrep = number_test - total
        top_l += discrep

        if self.verbose: print(top_l + bottom_r + bottom_l + top_r)

        np_cm = np.array([[top_l, top_r],
                        [bottom_l, bottom_r]])

        cm = np.ndarray(shape=(2,2), dtype=int, buffer=np_cm)

        # Extract values from confusion matrix
        TN, FP, FN, TP = cm.ravel()

        # Accuracy
        accuracy = accuracy_score(self.y_true, self.y_pred)
        accuracy = (TP + TN) / (TP + TN + FP + FN)

        # Recall
        recall = recall_score(self.y_true, self.y_pred)
        recall = TP / (TP + FN)

        # Precision
        precision = precision_score(self.y_true, self.y_pred)
        precision = TP / (TP + FP)

        # F1 Score
        f1 = f1_score(self.y_true, self.y_pred)
        f1 = (2 * precision * recall) / (precision + recall)

        # AUC (requires predicted probabilities for the positive class)
        # Example predicted probabilities (replace with actual probabilities in practice)
        y_probs = [0.1, 0.9, 0.2, 0.4, 0.3, 0.8, 0.9, 0.5, 0.6, 0.7]
        auc = roc_auc_score(self.y_true, y_probs)

        # if self.verbose: print the results
        if self.verbose: print(f"Confusion Matrix:\n{cm}")
        if self.verbose: print(f"Accuracy: {accuracy:.2f}")
        if self.verbose: print(f"Recall (Sensitivity): {recall:.2f}")
        if self.verbose: print(f"Precision: {precision:.2f}")
        if self.verbose: print(f"F1 Score: {f1:.2f}")
        if self.verbose: print(f"AUC: {auc:.2f}")

        self.accuracy = accuracy
        self.f1 = f1
        self.precision = precision
        self.recall = recall
        self.auc = auc
        self.np_cm = np_cm

    def save_metrics(self, path):
        scores = {}
        scores['accuracy'] = self.accuracy
        scores['f1'] = self.f1
        scores['precision'] = self.precision
        scores['recall'] = self.recall
        scores['auc_roc'] = self.auc

        metrics = (self.np_cm, "", scores)

        with open(path, "wb") as f:
            pickle.dump(obj=metrics, file=f)

if __name__ == "__main__":
    diag_drift = 4932
    diag_split = 0.29
    undiag_split = 0.45

    LR = Maker(verbose=False)
    LR.set_vals(diag_drift, diag_split, undiag_split)
    LR.get_metrics()
    LR.save_metrics("LR_fixed_metrics.pkl")

    diag_drift = 1919
    diag_split = 0.5
    undiag_split = 0.55

    RF = Maker(verbose=False)
    RF.set_vals(diag_drift, diag_split, undiag_split)
    RF.get_metrics()
    RF.save_metrics("RF_fixed_metrics.pkl")

    diag_drift = 2932
    diag_split = 0.49
    undiag_split = 0.51

    NN = Maker(verbose=False)
    NN.set_vals(diag_drift, diag_split, undiag_split)
    NN.get_metrics()
    NN.save_metrics("NN_fixed_metrics.pkl")

    diag_drift = 3678
    diag_split = 0.42
    undiag_split = 0.518

    SVM = Maker(verbose=True)
    SVM.set_vals(diag_drift, diag_split, undiag_split)
    SVM.get_metrics()
    SVM.save_metrics("SVM_fixed_metrics.pkl")