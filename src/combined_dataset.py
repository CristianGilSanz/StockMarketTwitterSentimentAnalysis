import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import yfinance as yf


def combine_stock_market_and_tweets():
    # Get all of the stock files to process
    files = os.listdir('../data/')

    # Initialize a new dataframe to hold stock data
    stocks = pd.DataFrame()

    # For each stock to process
    for stock in range(len(files))[0:1]:
        # Some tickers may be delisted, use try/except to avoid errors if the ticker does not exist in yFinance
        try:
            # Assign the current file name
            filename = files[stock]

            # Read the CSV file for the current ticker
            data = pd.read_csv('../data/' + filename + '/stock_data_sentiment.csv')

            # Convert string time to datetime
            data['Datetime'] = pd.to_datetime(data['Datetime'])

            # Convert 0 sentiment to -1
            data.loc[data['Sentiment'] == 0, 'Sentiment'] = -1

            # Initialize Weights and Tweet numbers
            data['Tweets'] = 1
            data['Weight'] = 1

            # Calculate weights based on user followers and retweets
            data = calculate_user_weights(data, 'Followers', 'Weight')
            data = calculate_user_weights(data, 'RTs', 'Weight')

            # Multiply the sentiment score by the individual weight
            data['Sentiment_Weighted'] = data['Sentiment'] * data['Weight']

            # Group the stock data by months and days
            data = data.groupby([data.Datetime.dt.month, data.Datetime.dt.day]).sum()

            # Reassign the ticker name that was lost after grouping
            data['Ticker'] = filename.split('_')[0]

            # Reassign date based on index values of month and day
            data['Date'] = pd.to_datetime(
                [str(stock) + '/' + str(y) + '/2016' for (stock, y) in data.index.values]) + datetime.timedelta(days=1)

            # Drop the index
            data = data.reset_index(drop=True)

            # Divide the sentiment by the total number of tweets
            data['Sentiment_Weighted'] /= data['Tweets']

            # Get the rolling average of the sentiment and tweet volume
            data['Sentiment_MA'] = data['Sentiment_Weighted'].rolling(3, min_periods=1).mean()
            data['Tweets_MA'] = data['Tweets'].rolling(3, min_periods=1).mean()

            # Get the starting date and ending date to extract stock prices
            start_date = data['Date'].min()
            end_date = data['Date'].max() + datetime.timedelta(days=2)

            # Download stock price data for the given range of tweets
            prices = download_and_calculate_price_changes(filename, start_date, end_date)

            # Combine the stock sentiment data and the pricing data
            data = data.merge(prices, on='Date', how='left')

            # Select relevant columns
            data = data[['Ticker', 'Date', 'Sentiment_Weighted', 'Sentiment_MA', 'Tweets', 'Tweets_MA', 'Adj Close',
                         'Percent_Change', 'Percent_Change_Bin']]

            # Drop missing values for days without pricing information
            data = data.dropna().reset_index(drop=True)

            # Save stock sentiment and pricing data to CSV
            data.to_csv('../data/' + filename + '/stock_data_inputs.csv', index=False)

            # Add ticker data to the main data set
            stocks = pd.concat([stocks, data])

            # Print status of ticker
            print(filename.split('_')[0], '- Completed')

        # If the ticker is not in yFinance
        except Exception as e:
            # Print error message
            print(filename.split('_')[0], '-', e)

    # Save data for all stocks to CSV
    stocks.to_csv('../outputs/combined_stock_inputs_AAL.csv', index=False)

def calculate_user_weights(data, user_column, weight_column):
    # Determine the mean and standard deviation of the number of followers/retweets
    # This ensures tweets with fewer tweets per day are treated equally
    data[f'{user_column}_Mean'] = data[user_column].rolling(10000, min_periods=1).mean()
    data[f'{user_column}_Std'] = data[user_column].rolling(10000, min_periods=1).std().fillna(data[user_column].std())

    # Weight tweets from users with higher follower counts/retweets more heavily
    data.loc[(data[user_column] >= data[f'{user_column}_Mean']) & (
                data[user_column] < (data[f'{user_column}_Mean'] + data[f'{user_column}_Std'])), weight_column] += 1
    data.loc[(data[user_column] >= (data[f'{user_column}_Mean'] + data[f'{user_column}_Std'])) & (
                data[user_column] < (data[f'{user_column}_Mean'] + data[f'{user_column}_Std'] * 2)), weight_column] += 2
    data.loc[data[user_column] >= (data[f'{user_column}_Mean'] + data[f'{user_column}_Std'] * 2), weight_column] += 3

    return data

def download_and_calculate_price_changes(filename, start_date, end_date):
    # Download stock price data for given range of tweets
    prices = yf.download(tickers=filename.split('_')[0], start=start_date, end=end_date).reset_index()

    # Calculate percent change based on stock price changes
    prices['Percent_Change'] = (prices['Adj Close'].pct_change() * 100).shift(-1)

    # Bin percent changes by amount lost/gained by stock
    prices['Percent_Change_Bin'] = pd.cut(prices['Percent_Change'], [-100, 0, 2, 100], labels=[0, 1, 2])

    return prices
