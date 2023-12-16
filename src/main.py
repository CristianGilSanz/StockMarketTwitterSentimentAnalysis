import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sentiment_data_cleaner import *
from VADER_model import *
from Naive_Bayes_TF_IDF_model import *
from BERT_model import *
from BERT_sentiment_classification import *
from combined_dataset import *
from investment_algorithm import *

unprocessed_data = load_data("../training_dataset/stock_twitter_sentiment_scores.csv")

processed_data = clean_data(unprocessed_data)

#filter_data_rows(processed_data,"Preprocessed_Text","")

#If this is the first time you run the program, activate this block of code
#import nltk
#nltk.download('vader_lexicon')
#nltk.download('stopwords')

data = clean_stopwords(processed_data)

#VADER_model(data)

#Naive_Bayes_TF_IDF_model(data)

#BERT_model(data)

#BERT_sentiment_assignment()

#combine_stock_market_and_tweets()

apply_investment_algorithm()