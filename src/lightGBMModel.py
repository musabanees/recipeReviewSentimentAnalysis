from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, make_scorer

class LightGBMModel:
    def __init__( self ):
        """
        Initializes the LightGBM model with specified hyperparameters.
        """
        self.nestimators = 50
        self.learningRate = 0.1 
        self.maxDepth = -1
        self.classWeight = {0: 3.0, 1:2.0}
        self.randomState = 42

        print("***TESTING WEIGHTS: " , self.classWeight)

        self.model = LGBMClassifier(
            random_state = self.randomState,
            n_estimators = self.nestimators,
            learning_rate = self.learningRate,
            max_depth = self.maxDepth,
            class_weight = self.classWeight,
        )

        self.kFold = StratifiedKFold(n_splits=5, 
                                     shuffle=True, 
                                     random_state = self.randomState)
        self.f1Scorer = make_scorer(f1_score, average='binary')  

    def crossValidation( self, X_train, y_train ):
        """
        Performs K-Fold cross-validation and returns the F1-scores for each fold.

        Args:
            X: Features for cross-validation.
            y: Labels for cross-validation.

        Returns:
            list: F1-scores for each fold.
        """
        f1Scorevalues = cross_val_score(self.model, 
                                        X_train,
                                        y_train, 
                                        cv = self.kFold, 
                                        scoring=self.f1Scorer)
        
        print(f"Mean F1-Score: {f1Scorevalues.mean():.2f}")
        print(f"Standard Deviation of F1-Score: {f1Scorevalues.std():.2f}")

    def train(self, X_train, y_train):
        """
        Trains the LightGBM model on the provided training data.

        Args:
            X_train: Training features.
            y_train: Training labels.
        """
        self.model.fit(X_train, y_train)
        print(" ** Model has Trained! ** ")


    # def evaluateModel(self, X_test, y_test):
    #     """
    #     Evaluates the trained model on the test data and prints performance metrics.

    #     Args:
    #         X_test: Test features.
    #         y_test: Test labels.

    #     Returns:
    #         dict: Evaluation metrics including precision, recall, F1-score.
    #     """
    #     y_pred = self.model.predict(X_test)
    #     test_f1 = f1_score(y_test, 
    #                        y_pred, 
    #                        average='binary')
        
    #     print(f" ** F1 Score on Test Data: {test_f1:.2f} **")
    #     print("\n** Classification Report: **")
    #     print(classification_report(y_test, y_pred))

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