import re
import os
import nltk
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import AllKNN

from sklearn.metrics import confusion_matrix , f1_score , classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from logisticRegressionModel import LogisticRegressionModel
from lightGBMModel import LightGBMModel

import tensorflow as tf

def read_dataset():
    """
        Reading the dataset and returning.
    """
    data_path = os.path.join('..', 'data', 'review.csv')
    data = pd.read_csv(data_path)
    return data

def PreprocessingText( text ):
    """
    Preprocesses the input text by performing the following steps:
    - Removes HTML tags and entities.
    - Removes special characters, punctuation, and digits.
    - Converts the text to lowercase.
    - Tokenizes the text into words.
    - Removes stopwords using the NLTK stopword list.
    - Applies lemmatization to each word using the WordNet Lemmatizer.

    Args:
        text (str): The input text to preprocess.

    Returns:
        str: The preprocessed text as a single string of lemmatized words, 
             or the original input if it is not a string.
    """
    stopWordsList = set(stopwords.words('english'))
    lemmatizerObj = WordNetLemmatizer()

    if isinstance(text, str): 
        text = re.sub(r'&\w+;', ' ', text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = text.lower()

        wordsbank = nltk.word_tokenize(text)
        words = [word for word in wordsbank if word not in stopWordsList]
        lemmatizedWords = [lemmatizerObj.lemmatize(word, pos="v") for word in words] 

        return " ".join(lemmatizedWords)
    return text

def dataCleaning( data ):
    """
    Cleans and preprocesses the input dataset by:
    - Removing rows where the 'text' column contains null values.
    - Filtering out rows where the 'stars' column equals zero.
    - Creating a new 'sentiment' column:
      - Assigns 0 for 'stars' values less than or equal to 3.
      - Assigns 1 for 'stars' values greater than 3.

    Args:
        data (pd.DataFrame): The input dataset containing 'text' and 'stars' columns.

    Returns:
        pd.DataFrame: The cleaned and processed dataset with a new 'sentiment' column.
    """
    print("Applying Text Processing and Data Cleaning!")
    data['cleaned_text'] = data['text'].apply(PreprocessingText)
    data = data.dropna(subset=['text'])
    data = data[data['stars'] != 0]
    data['sentiment'] = data['stars'].apply(lambda x: 0 if x <= 3 else 1)

    drop_columns = ['Unnamed: 0', 'recipe_number', 'recipe_code', 'recipe_name',
       'comment_id', 'user_id', 'user_name', 'user_reputation', 'created_at',
       'reply_count', 'thumbs_up', 'thumbs_down' , 'best_score']
    data = data.drop(columns=drop_columns , axis = 1)

    return data

def displayRatings( data ):
    """
    Displays a bar chart of sentiment class distribution from the dataset.

    The method:
    - Counts the occurrences of each sentiment class (0 for negative, 1 for positive) 
      in the 'sentiment' column of the dataset.
    - Plots the sentiment distribution as a bar chart with:
      - Red for negative sentiment.
      - Green for positive sentiment.
    - Includes labels and a title for better visualization.

    Args:
        data (pd.DataFrame): The input dataset containing a 'sentiment' column.

    Returns:
        None: Displays the bar chart and does not return any value.
    """

    sentiment_counts = data.value_counts()

    plt.figure(figsize=(8, 6))
    sentiment_counts.plot(kind='bar', color=['red', 'green'])
    plt.title("Sentiment Class Distribution")
    plt.xlabel("Sentiment Class (0 = Negative, 1 = Positive)")
    plt.ylabel("Number of Reviews")
    plt.xticks(rotation=0)
    plt.show()

def balanceClass( data, targetRating = 'sentiment', factor = 8000 ):
    """
    Balances the dataset by reducing the majority class size.

    Args:
        data (pd.DataFrame): Dataset containing the target rating column.
        targetRating (str): Column name for classification target. Default is 'sentiment'.
        factor (int): Number of rows to retain from the positive class. Default is 8000.

    Returns:
        pd.DataFrame: Balanced dataset with majority class reduced and rows shuffled.
    """
    
    positiveRows = data[data[targetRating] == 1]  
    negativeRows = data[data[targetRating] == 0] 

    positiveRowsSample = positiveRows.sample( n = factor,
                                                 random_state=42)  
    positiveRows = positiveRows.drop(positiveRowsSample.index) 

    data = pd.concat([positiveRows, negativeRows])  
    data = data.sample( frac=1, 
                        random_state=42).reset_index(drop=True) 

    data = data.dropna(subset=['cleaned_text'])

    return data

def allknnAlgo( X_train, y_train ):
    """
    Applies the All-KNN algorithm for undersampling the training data.

    Args:
        X_train (pd.DataFrame or np.ndarray): Features of the training dataset.
        y_train (pd.Series or np.ndarray): Target labels of the training dataset.

    Returns:
        None: Prints the class distribution after All-KNN undersampling.
    """
    allKnn = AllKNN( allow_minority=False )
    X_trainAllknn, y_trainAllknn = allKnn.fit_resample(X_train, y_train)

    print("Class distribution after All-KNN undersampling:", Counter(y_trainAllknn))
    return X_trainAllknn, y_trainAllknn

def smoteAlgo( X_train, y_train ):
    """
    Apply SMOTE for oversampling the minority class.
    
    Parameters:
        X_train: Features of the training set (sparse or dense matrix).
        y_train: Labels of the training set.
    
    Returns:
        X_resampled: Resampled features.
        y_resampled: Resampled labels.
    """
    smoteObj = SMOTE(sampling_strategy={0: y_train.value_counts()[1]}, 
                                        random_state=42)
    xSmoteData, ySmoteData = smoteObj.fit_resample(X_train, y_train)
    
    print("Class distribution after SMOTE:", Counter(ySmoteData))
    
    return xSmoteData, ySmoteData

def adasynAlgo(X_train, y_train):
    """
    Applies the ADASYN algorithm for oversampling the training data.

    Args:
        X_train (pd.DataFrame or np.ndarray): Training features.
        y_train (pd.Series or np.ndarray): Training labels.

    Returns:
        tuple: Resampled features and labels after ADASYN.
    """
    adasynObj = ADASYN(random_state = 777, 
                    sampling_strategy = 1.0)
    xadasynData, yadasynData = adasynObj.fit_resample(X_train, y_train)
    
    print("Class distribution after ADASYN:", Counter(yadasynData))
    
    return xadasynData, yadasynData

def trainBaselineModel(baselineClassifier, vectorTrain, trainLabel, vectorTest , testLabel, plotLabel = "Confusion Matrix"):
    """
    Trains a classifier and evaluates its performance.

    Args:
        baselineClassifier: Model to be trained and tested.
        vectorTrain: Training feature vectors.
        trainLabel: Labels for training data.
        vectorTest: Testing feature vectors.
        testLabel: Labels for testing data.
        plotLabel (str): Title for the confusion matrix plot. Default is "Confusion Matrix".

    Returns:
        None: Prints F1-scores and displays the confusion matrix heatmap.
    """
    baselineClassifier.fit(vectorTrain, trainLabel)
    
    ytrainPred = baselineClassifier.predict(vectorTrain)  
    f1Train = f1_score(trainLabel, ytrainPred)

    yTestPred = baselineClassifier.predict(vectorTest)
    f1Test = f1_score(testLabel, yTestPred)

    print(f"Training F1-Score: {f1Train:.4f}")
    print(f"Testing F1-Score: {f1Test:.4f}")

    cMap = confusion_matrix(testLabel, yTestPred)
    sns.heatmap(cMap, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(plotLabel)
    plt.show()


def tfidfVectorizationProcess( Xdata , DataLabels ):
    """
    Splits the data and applies TF-IDF vectorization.

    Args:
        Xdata (iterable): Text data for vectorization.
        DataLabels (iterable): Target Labels corresponding to the data.

    Returns:
        tuple: TF-IDF transformed training and test sets (X_train, X_test, y_train, y_test).
    """
    X_train, X_test, y_train, y_test = train_test_split( Xdata, 
                                                         DataLabels, 
                                                         test_size=0.3, 
                                                         random_state=42)

    tfidfVectorizer = TfidfVectorizer( ngram_range=(1, 3), 
                                        max_df=0.75, 
                                        min_df=10, 
                                        sublinear_tf=True)

    X_train = tfidfVectorizer.fit_transform( X_train )
    X_test = tfidfVectorizer.transform( X_test )
    print("Shape After Vectorization! ")
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    return X_train, X_test, y_train, y_test , tfidfVectorizer


def runModels( xSmoteData, ySmoteData, X_test, y_test ):
    """
    Trains and evaluates a logistic regression model.

    Args:
        xSmoteData: Training feature set after SMOTE.
        ySmoteData: Training labels after SMOTE.
        X_test: Test feature set.
        y_test: Test labels.

    Returns:
        float: Final F1 score of the logistic regression model.
    """

    logisticRegressionModel = LogisticRegressionModel()
    logisticRegressionModel.crossValidation(xSmoteData, ySmoteData)
    logisticRegressionModel.train(xSmoteData, ySmoteData)
    # logisticRegressionModel.evaluateModel(X_test, y_test)


    lightGBMModel = LightGBMModel()
    lightGBMModel.crossValidation(xSmoteData, ySmoteData)
    lightGBMModel.train(xSmoteData, ySmoteData)
    # lightGBMModel.evaluateModel(X_test, y_test)

    return logisticRegressionModel , lightGBMModel


def evaluateModel( model , X_test, y_test ):

    yTestPred , _ = model.predict( X_test )
    testF1Score = f1_score(y_test, yTestPred, average='binary')
    print(f"** F1-Score on Test Data: {testF1Score:.2f} **")

    print("\n** Classification Report: **")
    print(classification_report(y_test, yTestPred))

    print("\n** Confusion Matrix **")
    cMap = confusion_matrix(y_test, yTestPred)
    sns.heatmap(cMap, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title('Confusion Matrix')
    plt.show()

def predictSentiment( text , tfidfVectorizer, model ):

    tfidfVector = tfidfVectorizer.transform([text])

    y_pred , y_pred_proba = model.predict(tfidfVector)
    sentimentDic = {1: "Positive" , 0: "Negavtive"}

    print(f"Predicted Label: {sentimentDic[y_pred[0]]}")
    print(f"Probability of Positive Class: {y_pred_proba}")
