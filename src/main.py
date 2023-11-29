import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sentiment_data_cleaner import *
from sentiment_classification import *

unprocessed_data = load_data("../data/stock_twitter_sentiment_scores.csv")

processed_data = clean_data(unprocessed_data)

#filter_data_rows(processed_data,"Preprocessed_Text","")


#If this is the first time you run the program, activate this block of code
import nltk
#nltk.download('vader_lexicon')
#nltk.download('stopwords')

data = calc_sentyment_polarity_from_dataset(processed_data, "Preprocessed_Text")

train_model(data)

