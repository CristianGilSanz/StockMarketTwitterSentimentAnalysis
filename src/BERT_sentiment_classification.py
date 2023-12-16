import os
from BERT_classifier import *
import pandas as pd
from sentiment_data_cleaner import *
from transformers import BertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler


def BERT_sentiment_assignment():
    # Load the pre-trained BERT sentiment analysis model
    bert_model = BertClassifier()
    bert_model.load_state_dict(torch.load("../outputs/stock_sentiment_model.pt"))

    # Get the list of files, each representing a stock, in the data directory
    stock_files = os.listdir("../data/")

    # Iterate through each stock file
    for stock_index in range(len(stock_files))[0:1]:
        # Read the Excel file from the 'Stream' sheet for the current stock
        stock_data = pd.read_excel(
            '../data/' + stock_files[stock_index] + '/export_dashboard_' + stock_files[stock_index],
            sheet_name='Stream')

        # Assign the stock ticker as a column
        stock_data['Ticker'] = stock_files[stock_index].split('_')[0]

        # Convert string date times to datetime
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        stock_data['Hour'] = stock_data['Hour'].apply(lambda t: pd.Timedelta(hours=int(t[:2]), minutes=int(t[3:])))
        stock_data['Datetime'] = stock_data['Date'] + stock_data['Hour']

        # Rename the column holding the tweet content
        stock_data.rename(columns={'Tweet content': 'Text'}, inplace=True)

        # Pre-process the tweet content
        stock_data = clean_data(stock_data)
        stock_data = clean_stopwords(stock_data)

        # Keep relevant columns
        stock_data = stock_data[
            ['Tweet Id', 'Ticker', 'Datetime', 'Text', 'Preprocessed_Text_No_Stopwords', 'Favs', 'RTs', 'Followers',
             'Following', 'Is a RT']]

        # Fill NAs in Favs, RTs, Followers, and Following with 0
        stock_data = stock_data.fillna(0)

        # Initialize BERT tokenizer
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        # Initialize lists to store BERT input features
        input_ids_list = []
        attention_masks_list = []

        # Tokenize and encode each tweet's text
        for tweet_text in stock_data['Preprocessed_Text_No_Stopwords'].values:
            encoding = bert_tokenizer.encode_plus(
                text=tweet_text,
                add_special_tokens=True,
                padding='max_length',
                max_length=48,
                truncation=True,
                return_tensors="pt",
                return_attention_mask=True
            )

            # Add the encodings to the lists
            input_ids_list.append(encoding.get('input_ids'))
            attention_masks_list.append(encoding.get('attention_mask'))

        # Combine BERT input features
        stock_input_ids = torch.cat(input_ids_list)
        stock_attention_masks = torch.cat(attention_masks_list)

        # Check if GPU is available and assign device
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        # Put stock data in PyTorch DataLoader for processing
        stock_dataset = TensorDataset(stock_input_ids, stock_attention_masks)
        stock_sampler = RandomSampler(stock_dataset)
        stock_dataloader = DataLoader(stock_dataset, sampler=stock_sampler, batch_size=16)

        # Set the BERT model to evaluation mode
        bert_model.eval()

        # Initialize list to store BERT predictions
        bert_predictions = []

        # For each batch
        for batch in stock_dataloader:
            # Get encoded inputs and masks
            batch_input_ids, batch_attention_masks = batch

            # Send variables to device (GPU if available)
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_masks = batch_attention_masks.to(device)

            # Predict sentiment with the BERT model for given inputs
            with torch.no_grad():
                logits = bert_model(batch_input_ids, batch_attention_masks)

            # Convert logits to 0s and 1s (class predictions)
            batch_predictions = torch.argmax(logits, dim=1).flatten()
            bert_predictions.append(batch_predictions)

        # Combine all batch predictions
        bert_predictions = torch.cat(bert_predictions).cpu().numpy()

        # Add BERT predictions to the stock dataframe
        stock_data['Sentiment'] = bert_predictions

        # Save the stock dataframe with sentiment predictions as a new CSV
        stock_data.to_csv('../data/' + stock_files[stock_index] + '/stock_data_sentiment.csv', index=False)

        # Show the completion message for the current stock
        print(stock_files[stock_index].split('_')[0], '- completed')