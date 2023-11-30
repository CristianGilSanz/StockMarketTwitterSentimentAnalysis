from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB

import matplotlib.pyplot as plt
import numpy as np


def Naive_Bayes_TF_IDF_model(data):

    # This section splits the dataset into training and test sets. It shuffles the indices to randomize the data, then
    # selects a portion of the data for training and the remaining portion for testing.

    # Split data into training and test sets (80/20)
    training_percentage = 0.8
    random_seed = 1
    np.random.seed(random_seed)

    # Get shuffled indices
    shuffled_indices = np.random.permutation(len(data))

    # Training set
    training_texts = data['Preprocessed_Text_No_Stopwords'].values[
        shuffled_indices[:int(training_percentage * len(data))]]
    training_labels = data['Sentiment'].values[shuffled_indices[:int(training_percentage * len(data))]]
    # The sentiment labels are adjusted from -1 to 0 for compatibility.
    training_labels[training_labels == -1] = 0

    # Test set
    test_texts = data['Preprocessed_Text_No_Stopwords'].values[shuffled_indices[int(training_percentage * len(data)):]]
    test_labels = data['Sentiment'].values[shuffled_indices[int(training_percentage * len(data)):]]
    test_labels[test_labels == -1] = 0

    # Calculate TF-IDF for Naive Bayes classification. The TF-IDF matrix is a representation of the original text data,
    # where each row corresponds to a tweet, and each column corresponds to a unique term. The values in the matrix
    # represent the TF-IDF scores for each term in each document.
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3),
                                       binary=True,
                                       smooth_idf=False)

    # Get TF-IDF for training and test data
    tfidf_matrix_train = tfidf_vectorizer.fit_transform(training_texts)
    tfidf_matrix_test = tfidf_vectorizer.transform(test_texts)

    # Initialize models with multiple alpha values to find the best model.The purpose of Laplace smoothing, or additive
    # smoothing, is to handle the issue of zero probabilities for certain features during the training of a Naive Bayes
    # classifier.
    alpha_values = np.arange(0.01, 10, 0.01)
    naive_bayes_models = [MultinomialNB(alpha=alpha) for alpha in alpha_values]

    # Find the best model using cross-validation.

    # StratifiedKFold is a variation of k-fold cross-validation that ensures the distribution of target classes is
    # maintained in each fold. This is particularly useful when dealing with imbalanced datasets where one class may be
    # significantly less frequent than the other.
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)

    # The ROC curve is a graphical representation of a model's performance across different classification thresholds.
    # It plots the True Positive Rate against the False Positive Rate  for various threshold values, in this case, alpha.

    auc_roc_scores = [cross_val_score(model, tfidf_matrix_train, training_labels, scoring="roc_auc", cv=kf).mean() for model
                      in naive_bayes_models]
    auc_roc_scores = np.array(auc_roc_scores)

    # Plot AUC_ROC scores for different alpha values
    plt.figure(figsize=(15, 7))
    plt.plot(alpha_values, auc_roc_scores)
    plt.xlabel('Alpha value')
    plt.ylabel('AUC-ROC Score')
    plt.title('AUC-ROC Score for Different Alpha Values')
    plt.show()

    # Get the alpha value that produces the best performance
    best_alpha = round(alpha_values[auc_roc_scores.argmax()], 1)

    # Retrain the best model with the entire training set
    best_naive_bayes_model = MultinomialNB(alpha=best_alpha)
    best_naive_bayes_model.fit(tfidf_matrix_train, training_labels)

    # Predict test data with the best model
    probabilities = best_naive_bayes_model.predict_proba(tfidf_matrix_test)

    # Evaluate and print the accuracy of the best Naive-Bayes model for sentiment analysis
    correct_predictions = np.sum(test_labels == np.argmax(probabilities, axis=1))
    total_predictions = len(probabilities)
    accuracy_best_naive_bayes_model = (correct_predictions / total_predictions) * 100

    print('Naive-Bayes Model Accuracy:', round(accuracy_best_naive_bayes_model, 2), '%')

    # Print the hyperparameter and performance metrics of the best Naive-Bayes model
    print('\t Best Alpha Value: ', best_alpha)
    print('\t Best AUC-ROC Score: ', round(auc_roc_scores.max(), 4))