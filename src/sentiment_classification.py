from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB

import matplotlib.pyplot as plt

import numpy as np

def clean_stopwords(data):
    stop_words = set([s.replace("'", '') for s in stopwords.words('english') if s not in ['not', 'up', 'down', 'above', 'below', 'under', 'over']])

    data['Preprocessed_Text_No_Stopwords'] = data['Preprocessed_Text'].apply(lambda s: " ".join([word for word in s.split() if word not in stop_words]))
    data['Preprocessed_Text_No_Stopwords'] = data['Preprocessed_Text_No_Stopwords'].str.strip()

    return data

def VADER_model(data):
    #Create an instance of the sentiment analyzer
    sid = SentimentIntensityAnalyzer()

    #Assign a polarity score to each tweet
    data['Sentiment_Polarity_Score'] = data["Preprocessed_Text"].apply(lambda score: sid.polarity_scores(score)['compound'])
    data['Rounded_Predicted_Score'] = data['Sentiment_Polarity_Score'].apply(lambda score: 1 if score >=0 else -1)

    #Calculate the accuracy of the sentiment analyzer
    correct_predictions = (data['Sentiment'] == data['Rounded_Predicted_Score']).sum()
    accuracy = (correct_predictions / len(data)) * 100

    print('VADER Model Accuracy:', round(accuracy, 2), '%', '\n')

def Naive_Bayes_TF_IDF_model(data):
    #Split data into training and test sets (80/20)
    training_percentage = 0.8
    random_seed = 1
    np.random.seed(random_seed)

    #Get shuffled indices
    shuffled_indices = np.random.permutation(len(data))

    #Training set
    training_texts = data['Preprocessed_Text_No_Stopwords'].values[
        shuffled_indices[:int(training_percentage * len(data))]]
    training_labels = data['Sentiment'].values[shuffled_indices[:int(training_percentage * len(data))]]
    training_labels[training_labels == -1] = 0

    #Test set
    test_texts = data['Preprocessed_Text_No_Stopwords'].values[shuffled_indices[int(training_percentage * len(data)):]]
    test_labels = data['Sentiment'].values[shuffled_indices[int(training_percentage * len(data)):]]
    test_labels[test_labels == -1] = 0

    #Calculate TF-IDF for Naive Bayes classification
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3),
                                       binary=True,
                                       smooth_idf=False)

    #Get TF-IDF for training and test data
    X_train_tfidf = tfidf_vectorizer.fit_transform(training_texts)
    X_test_tfidf = tfidf_vectorizer.transform(test_texts)

    #Initialize models with multiple alpha values to find the best model
    alpha_values = np.arange(1, 10, 0.1)
    naive_bayes_models = [MultinomialNB(alpha=alpha) for alpha in alpha_values]

    #Find the best model using cross-validation
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    auc_roc_scores = [cross_val_score(model, X_train_tfidf, training_labels, scoring="roc_auc", cv=kf).mean() for model
                      in naive_bayes_models]
    auc_roc_scores = np.array(auc_roc_scores)

    #Get the alpha value that produces the best performance
    best_alpha = round(alpha_values[auc_roc_scores.argmax()], 1)

    #Plot accuracies for different alpha values
    plt.figure(figsize=(15, 7))
    plt.plot(alpha_values, auc_roc_scores)
    plt.xlabel('Alpha value')
    plt.ylabel('AUC-ROC Score')
    plt.title('AUC-ROC Score for Different Alpha Values')
    plt.show()

    #Retrain the best model with the entire training set
    best_naive_bayes_model = MultinomialNB(alpha=best_alpha)
    best_naive_bayes_model.fit(X_train_tfidf, training_labels)

    #Predict test data with the best model
    probabilities = best_naive_bayes_model.predict_proba(X_test_tfidf)

    #Print accuracy of the best model on sentiment analysis
    accuracy_best_naive_bayes_model = (test_labels == np.argmax(probabilities, axis=1)).sum() / len(probabilities) * 100
    print('Naive-Bayes Model Accuracy:', round(accuracy_best_naive_bayes_model, 2), '%')

    # Print the best alpha value and accuracy
    print('\t Best alpha value: ', best_alpha, '\n \t Best AUC-ROC Score: ', round(auc_roc_scores.max(), 4))