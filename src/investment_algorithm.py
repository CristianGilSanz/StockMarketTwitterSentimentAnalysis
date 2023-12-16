import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import warnings
from sklearn.ensemble import RandomForestClassifier

def apply_investment_algorithm():
    # Load combined sentiment and stock price data from a CSV file
    file_path = '../outputs/combined_stock_inputs.csv'
    stock_data = pd.read_csv(file_path, parse_dates=['Date'])

    # Sort values in stock sentiment/pricing data by date and ticker name
    stock_data = stock_data.sort_values(['Date', 'Ticker'])

    # Split data into training and testing sets
    train_data = stock_data.loc[stock_data['Date'] < datetime.datetime(year=2016, month=4, day=15)].reset_index(drop=True)
    test_data = stock_data.loc[stock_data['Date'] >= datetime.datetime(year=2016, month=4, day=15)].reset_index(drop=True)

    # Select features and labels for training data
    X_train = train_data[['Sentiment_Weighted', 'Sentiment_MA', 'Tweets', 'Tweets_MA']].values
    y_train = train_data['Percent_Change_Bin'].values

    # Select features and labels for test data
    X_test = test_data[['Sentiment_Weighted', 'Sentiment_MA', 'Tweets', 'Tweets_MA']].values
    y_test = test_data['Percent_Change_Bin'].values

    # Define the random forest classifier
    model = RandomForestClassifier(random_state=1)

    # Train the model with training data
    model.fit(X_train, y_train)

    # Predict the test data
    predictions = model.predict(X_test)

    # Print the percentage of predictions that resulted in investing in losing stocks
    losing_investments_percentage = len(np.where((predictions > 0) & (y_test == 0))[0]) / len(predictions)
    print("Percentage of predicted losing investments:", losing_investments_percentage)
    print('\n')

    # Add predictions to the test dataset
    test_data['Prediction'] = predictions

    # Initialize initial capital to evaluate model effectiveness
    bot_initial_capital = 1000000
    long_initial_capital = 1000000

    # Add initial capital as the first data points
    bot_capitals = [bot_initial_capital]
    long_capitals = [long_initial_capital]

    # Get unique dates
    unique_dates = test_data['Date'].sort_values().unique()

    # For each unique date
    for date in unique_dates:
        # Filter the DataFrame for the specific date
        date_data = test_data[test_data['Date'] == date]

        # Calculate profit from investing equal parts in all stocks within the time frame
        long_initial_capital = long_initial_capital + (
                (long_initial_capital / len(date_data)) * (date_data['Percent_Change'] / 100)).sum()

        # Calculate profit from using the model to determine which stocks to invest in
        selected_stocks = date_data[date_data['Prediction'] > 0]
        bot_initial_capital = bot_initial_capital + (
                (bot_initial_capital / len(selected_stocks)) * (selected_stocks['Percent_Change'] / 100)).sum()

        # Keep track of the account totals over time
        long_capitals.append(long_initial_capital)
        bot_capitals.append(bot_initial_capital)

    # Filter deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Plot the account balances over time
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(unique_dates, long_capitals[1:], c='g', label='Long-Term Investment')
    ax.plot(unique_dates, bot_capitals[1:], c='b', label='Bot Investment')
    ax.legend()
    plt.show()

    # Show the return from each account over time
    print('Long-Term Investment:', round(long_capitals[-1], 2),
          '(', round((long_capitals[-1] - long_capitals[0]) / long_capitals[0] * 100, 2), '% )')
    print('Bot Trading:', round(bot_capitals[-1], 2),
          '(', round((bot_capitals[-1] - bot_capitals[0]) / bot_capitals[0] * 100, 2), '% )')
