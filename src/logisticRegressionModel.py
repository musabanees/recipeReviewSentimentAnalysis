import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, make_scorer , classification_report

class LogisticRegressionModel:
    def __init__(self, n_splits=5):
        """
        Initializes the Logistic Regression trainer with cross-validation setup.

        Parameters:
        - max_iter: Maximum number of iterations for Logistic Regression.
        - random_state: Random seed for reproducibility.
        - n_splits: Number of splits for Stratified K-Fold Cross-Validation.
        """
        self.max_iteration = 1000
        self.randomState = 42
        self.model = LogisticRegression(max_iter = self.max_iteration, 
                                        random_state = self.randomState,
                                        class_weight={0:2.0, 1:1.0})
        self.kfolds = StratifiedKFold(n_splits=n_splits, 
                                     shuffle=True, 
                                     random_state = self.randomState)
        
        self.f1ScorerMetric = make_scorer( f1_score, average='binary')

    def crossValidation(self, X_train, y_train):
        """
        Performs cross-validation on the training data.

        Parameters:
        - X_train: Training feature matrix.
        - y_train: Training target labels.

        Returns:
        - Mean F1-Score and standard deviation.
        """
        f1ScoresValues = cross_val_score( self.model, 
                                    X_train, 
                                    y_train, 
                                    cv=self.kfolds, 
                                    scoring = self.f1ScorerMetric)
        
        print(f"Mean F1-Score: {np.mean(f1ScoresValues):.4f}")
        print(f"Standard Deviation of F1-Score: {np.std(f1ScoresValues):.4f}")

    def train(self, X_train, y_train):
        """
        Trains the Logistic Regression model on the entire training set.

        Parameters:
        - X_train: Training feature matrix.
        - y_train: Training target labels.
        """
        self.model.fit(X_train, y_train)
        print(" ** Model has Trained! ** ")

    # def evaluateModel(self, X_test, y_test):
    #     """
    #     Evaluates the model on the test set.

    #     Parameters:
    #     - X_test: Test feature matrix.
    #     - y_test: Test target labels.

    #     Returns:
    #     - F1-Score on the test data.
    #     """
    #     y_pred = self.model.predict(X_test)
    #     testF1Score = f1_score(y_test, y_pred, average='binary')
    #     print(f"** F1-Score on Test Data: {testF1Score:.2f} **")
    #     print("\n** Classification Report: **")
    #     print(classification_report(y_test, y_pred))
    #     return testF1Score

    def predict( self, X_test ):
        """
        Makes predictions using the trained model.

        Args:
            X_test: Feature set for which predictions are to be made.

        Returns:
            tuple: Predicted class labels and their probabilities.
        """
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        return y_pred , y_pred_proba