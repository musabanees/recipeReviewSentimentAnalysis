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
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.metrics import make_scorer, recall_score

from sklearn.metrics import confusion_matrix , f1_score , classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

from logisticRegressionModel import LogisticRegressionModel
from lightGBMModel import LightGBMModel

import tensorflow as tf

import scipy

from collections import Counter
from imblearn.over_sampling import ADASYN
import numpy as np

def readDataset():
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

def adasynAlgo(X_train, y_train):

    # if scipy.sparse.issparse(X_train):
    #     X_train = X_train.toarray()

    # y_train = np.ravel(y_train)

    targetClass = Counter(y_train)
    
    # if len(distClass) != 2:
    #     raise ValueError("ADASYN requires binary classification. Found {} classes.".format(len(distClass)))

    majorityClassHighCount = max(targetClass, key = targetClass.get)
    minorityClassHighCount = min(targetClass, key = targetClass.get)
    
    SampleDifference = targetClass[majorityClassHighCount] - targetClass[minorityClassHighCount]

    adasyn = ADASYN(
        n_neighbors = 2,
        sampling_strategy = {minorityClassHighCount: SampleDifference},
        random_state = 42
    )

    XSampledData, ySampledData = adasyn.fit_resample(X_train, y_train)

    print(f"Original shape: {X_train.shape}, Resampled shape: {XSampledData.shape}")
    print(f"Original class distribution: {Counter(y_train)}")
    print(f"Resampled class distribution: {Counter(ySampledData)}")

    return XSampledData, ySampledData

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
                                                         test_size=0.2, 
                                                         random_state=42)

    tfidfVectorizer = TfidfVectorizer( ngram_range=(1, 2), 
                                        max_df=0.6, 
                                        min_df=0.01, 
                                        sublinear_tf=True)
    
    

    X_train = tfidfVectorizer.fit_transform( X_train )
    X_test = tfidfVectorizer.transform( X_test )

    print("Shape After Vectorization! ")
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    return X_train, X_test, y_train, y_test , tfidfVectorizer


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

    
    print("Class distribution Before SMOTE:", Counter(y_train))    
    print("Class distribution after SMOTE:", Counter(ySmoteData))
    
    return xSmoteData, ySmoteData


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


    lightGBMModel = LightGBMModel()
    lightGBMModel.crossValidation(xSmoteData, ySmoteData)
    lightGBMModel.train(xSmoteData, ySmoteData)

    return logisticRegressionModel , lightGBMModel


def evaluateModel( model , X_test, y_test ):

    yTestPred , _ = model.predict( X_test )
    testF1Score = f1_score(y_test, yTestPred, average='binary')
    f1Scores = model.returnf1CrossValidationScore()
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


    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(f1Scores) + 1), f1Scores, 
            color='skyblue', edgecolor='navy', alpha=0.7)
    
    meanScore = np.mean(f1Scores)
    plt.axhline(y = meanScore, color='red', linestyle='--', 
                label=f'Mean F1-Score: {meanScore:.2f}')
    
    plt.title('Cross-Validation F1 Scores', fontsize=16)
    plt.xlabel('Fold Number', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.ylim(0, 1)  
    
    for i, score in enumerate(f1Scores):
        plt.text(i + 1, score, f'{score:.3f}', 
                 ha='center', va='bottom')
    plt.legend()
    plt.tight_layout()
    plt.show()


def predictSentiment( text , tfidfVectorizer, model ):

    tfidfVector = tfidfVectorizer.transform([text])

    y_pred , y_pred_proba = model.predict(tfidfVector)
    sentimentDic = {1: "Positive" , 0: "Negavtive"}

    print(f"Predicted Label: {sentimentDic[y_pred[0]]}")
    print(f"Probability of Positive Class: {y_pred_proba}")


def show_word_contributions(text, tfidfVectorizer, model, top_n=10):
    """
    Display how unigrams and bigrams contribute to the model's decision for a given text.
    
    Args:
        text (str): The input text to analyze.
        tfidfVectorizer: The trained TfidfVectorizer.
        model: The trained model (e.g., Logistic Regression or LightGBM).
        top_n (int): Number of top contributing words/n-grams to display.
    """
    textVector = tfidfVectorizer.transform([text]).toarray()[0]
    featureNames = tfidfVectorizer.get_feature_names_out()
    
    if hasattr(model, "coef_"):
        getModelCoefficients = model.coef_[0]
        wordContributions = [(featureNames[i], textVector[i] * getModelCoefficients[i]) 
                              for i in range(len(featureNames)) if textVector[i] > 0]
    elif hasattr(model, "feature_importances_"):
        getFeatureImportances = model.feature_importances_
        wordContributions = [(featureNames[i], textVector[i] * getFeatureImportances[i]) 
                              for i in range(len(featureNames)) if textVector[i] > 0]
    else:
        print("Wrong Model Object Provided!")
        return
    
    wordContributions = sorted(wordContributions, key=lambda x: abs(x[1]), reverse=True)
    
    print(f"\nTop {top_n} Contributions for Text:")
    for terms, contribution in wordContributions[:top_n]:
        print(f"{terms}: {contribution:.3f}")

    terms, scores = zip(*wordContributions[:top_n])
    plt.figure(figsize=(10, 8))
    plt.barh(terms, scores, color='blue')
    plt.xlabel("Contribution to Prediction")
    plt.title("Word Contributions to Prediction")
    plt.gca().invert_yaxis()
    plt.show()

