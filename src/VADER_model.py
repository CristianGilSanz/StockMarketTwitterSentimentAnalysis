from nltk.sentiment.vader import SentimentIntensityAnalyzer

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
