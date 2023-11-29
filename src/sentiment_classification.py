from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB

import matplotlib.pyplot as plt

import numpy

def clean_stopwords(data):
    stop_words = set([s.replace("'", '') for s in stopwords.words('english') if s not in ['not', 'up', 'down', 'above', 'below', 'under', 'over']])

    data['Preprocessed_Text_No_Stopwords'] = data['Preprocessed_Text'].apply(lambda s: " ".join([word for word in s.split() if word not in stop_words]))
    data['Preprocessed_Text_No_Stopwords'] = data['Preprocessed_Text_No_Stopwords'].str.strip()

    return data

def calc_sentyment_polarity_from_dataset(data, source_column):
    data = clean_stopwords(data)

    #Create an instance of the sentiment analyzer
    sid = SentimentIntensityAnalyzer()

    #Assign a polarity score to each tweet
    data['Sentiment_Polarity_Score'] = data[source_column].apply(lambda score: sid.polarity_scores(score)['compound'])
    data['Rounded_Predicted_Score'] = data['Sentiment_Polarity_Score'].apply(lambda score: 1 if score >=0 else -1)

    #Calculate the accuracy of the sentiment analyzer
    correct_predictions = (data['Sentiment'] == data['Rounded_Predicted_Score']).sum()
    accuracy = (correct_predictions / len(data)) * 100

    print('Sentiment Analyzer Accuracy:', round(accuracy, 2), '%', '\n')

    return data

def train_model(data):

    # Split data into 80/20 train-test split
    train_pct = .8
    numpy.random.seed(1)
    idx = numpy.random.permutation(len(data))

    X_train = data['Preprocessed_Text_No_Stopwords'].values[idx[:int(train_pct * len(data))]]
    y_train = data['Sentiment'].values[idx[:int(train_pct * len(data))]]
    y_train[y_train == -1] = 0
    X_test = data['Preprocessed_Text_No_Stopwords'].values[idx[int(train_pct * len(data)):]]
    y_test = data['Sentiment'].values[idx[int(train_pct * len(data)):]]
    y_test[y_test == -1] = 0

    # Calculate TF-IDF for Naive Bayes classification
    tf_idf = TfidfVectorizer(ngram_range=(1, 3),
                             binary=True,
                             smooth_idf=False)

    # Get TF-IDF for Train and Test data
    X_train_tfidf = tf_idf.fit_transform(X_train)
    X_test_tfidf = tf_idf.transform(X_test)

    # Define function to determine accuracy of model
    def get_auc_CV(model):
        # Set KFold to shuffle data before the split
        kf = StratifiedKFold(5, shuffle=True, random_state=1)

        # Get AUC scores
        auc = cross_val_score(model, X_train_tfidf, y_train, scoring="roc_auc", cv=kf)

        return auc.mean()

    # Initialize models with multiple alpha values to find best model
    alphas = numpy.arange(1, 10, 0.1)
    models = [MultinomialNB(alpha=i) for i in alphas]

    # Find best performing model
    accs = []
    for model in models:
        accs.append(get_auc_CV(model))

    accs = numpy.array(accs)

    # Get best performing alpha value to continue with
    best_alpha = round(alphas[accs.argmax()], 1)

    # Print best alpha value and accuracy
    print('Best alpha: ', best_alpha, '  |  Best Score: ', round(accs.max() * 100, 2))

    # Plot accuracies per alpha values
    plt.figure(figsize=(15, 7))
    plt.plot(alphas, accs)
    plt.show()

    # Retrain best performing model
    best_model = MultinomialNB(alpha=best_alpha)
    best_model.fit(X_train_tfidf, y_train)

    # Predict test data with best model
    probs = best_model.predict_proba(X_test_tfidf)

    # Print accuracy of best performing model on tweet sentiment analysis
    print('Naive-Bayes Accuracy:', round(len(numpy.where(y_test == probs.argmax(axis=1))[0]) / len(probs) * 100, 2), '%')
