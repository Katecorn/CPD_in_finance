import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os
import pickle
from PyIF import te_compute as te
from tqdm import tqdm
import ruptures as rpt
from community import community_louvain

class CryptoPreproc:
    def __init__(self, data):
        self.data = data

    def check_missing_dates(self):
        df = self.data.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        date_range = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='D')
        missing_dates = date_range.difference(df['Date'])
        if missing_dates.empty:
            print("All dates are present.")
        else:
            print(f"Missing dates: {missing_dates}")

    def add_date_features(self):
        df = self.data.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['DayOfMonth'] = df['Date'].dt.day
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        df['MonthOfYear'] = df['Date'].dt.month
        self.data = df
        return df

    def returns_calc(self):
        df = self.data
        df_returns = df.drop(columns=['Date']).pct_change()
        df_returns['Date'] = df.Date
        df_returns = df_returns.dropna().reset_index(drop=True)
        return df_returns

    def calculate_overall_correlation(self, df):
        overall_correlation = df.drop(columns=['Date']).corr()
        return overall_correlation

    def visualize_overall_correlation(self, overall_correlation):
        plt.figure(figsize=(10, 10))
        plt.imshow(overall_correlation, cmap='coolwarm', interpolation='nearest')
        plt.colorbar()
        plt.title('Overall Correlation')
        plt.xticks(range(len(overall_correlation)), overall_correlation.columns, rotation=45)
        plt.yticks(range(len(overall_correlation)), overall_correlation.columns)
        plt.show()

    def calculate_rolling_correlation(self, df, window='30D'):
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df.dropna()
        rolling_correlation = df.rolling(window=window).corr()
        return rolling_correlation.reset_index()

    def visualize_rolling_correlation(self, rolling_correlation, ticker='BTC-USD', window='30D'):
        if 'level_1' in rolling_correlation.columns:
            tickers = rolling_correlation['level_1'].unique()
        else:
            print("The rolling correlation DataFrame does not have the expected structure.")
            return

        if ticker not in tickers:
            print(f"Ticker '{ticker}' not found in data.")
            return

        pairs = [(ticker, t) for t in tickers if t != ticker]
        pairs_to_plot = pairs[:10]

        plt.figure(figsize=(20, 3))
        for i, pair in enumerate(pairs_to_plot):
            crypto1, crypto2 = pair
            rolling_corr = rolling_correlation[rolling_correlation['level_1'] == crypto1]
            if crypto2 not in rolling_corr.columns:
                print(f"Ticker '{crypto2}' not found in correlation data.")
                continue
            plt.plot(rolling_corr['Date'], rolling_corr[crypto2], label=f'{crypto1}-{crypto2}')

        plt.title(f'Rolling Correlation ({window}) Between Cryptocurrencies')
        plt.xlabel('Date')
        plt.ylabel('Correlation')
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()

    def calculate_transfer_entropy(self, df, window=30, lag=1):
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        te_values = []
        source_target_pairs = []
        dates = []

        for i in tqdm(range(window, len(df)), desc="Calculating Transfer Entropy"):
            current_date = df.index[i]
            past_data = df.iloc[i-window:i]

            for source in df.columns:
                for target in df.columns:
                    if source != target:
                        te_value = te.te_compute(past_data[source].values, past_data[target].values, k=lag)
                        te_values.append(te_value)
                        source_target_pairs.append((source, target))
                        dates.append(current_date)

        te_df = pd.DataFrame(source_target_pairs, columns=['Source', 'Target'])
        te_df['Date'] = dates
        te_df['TransferEntropy'] = te_values
        return te_df

    def visualize_transfer_entropy(self, te_data, target_coin='BTC-USD', title="Transfer Entropy Over Time"):
        tickers = te_data[te_data.Target == target_coin].Source.unique()
        pairs = [(target_coin, t) for t in tickers if t != target_coin]
        pairs_to_plot = pairs[:10]
        
        
        plt.figure(figsize=(20, 5))
        

        for pair in pairs_to_plot:
            source, target = pair
            te_pair_data = te_data[(te_data['Source'] == source) & (te_data['Target'] == target)]
            plt.plot(te_pair_data['Date'], te_pair_data['TransferEntropy'], label=f'{source}->{target}')

        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Transfer Entropy')
        plt.legend()
        plt.grid(True)
        plt.show()

    def calculate_rolling_mean(self, df, target, window=7):
        roll_mean = df[target].rolling(window=window).mean().shift(window - 1)
        rolling_std = df[target].rolling(window=window).std().shift(window - 1)

        upper_bond = roll_mean + 1.96 * rolling_std
        lower_bond = roll_mean - 1.96 * rolling_std

        plt.figure(figsize=(10, 3))
        plt.plot(df['Date'], roll_mean, label=f'Rolling Mean (window={window})')
        plt.plot(df['Date'], upper_bond, "r--", label="Upper Bond / Lower Bond", alpha=0.7)
        plt.plot(df['Date'], lower_bond, "r--", alpha=0.7)
        plt.xlabel('Date')
        plt.xticks(rotation=45)
        plt.gca().set_xticks(df['Date'][::100])
        plt.ylabel(f'{target}')
        plt.title(f'Rolling Mean {target}')
        plt.legend(loc="upper left")
        plt.grid(True)
        plt.legend()
        plt.show()

    def double_exponential_smoothing(self, series, alpha, beta):
        level, trend = series[0], series[1] - series[0]
        smoothed = [series[0]]

        for value in series[1:]:
            last_level, level = level, alpha * value + (1 - alpha) * (level + trend)
            trend = beta * (level - last_level) + (1 - beta) * trend
            smoothed.append(level + trend)

        return smoothed

    def double_exp_plot(self, df, column, alphas, betas):
        plt.figure(figsize=(10, 5))
        plt.plot(df['Date'], df[column], label="Actual", color='gray')
        for alpha in alphas:
            for beta in betas:
                smoothed = self.double_exponential_smoothing(df[column], alpha, beta)
                plt.plot(df['Date'], smoothed, label="Alpha {}, beta {}".format(alpha, beta))
        plt.legend(loc="best")
        plt.axis('tight')
        plt.title(f"Double Exponential Smoothing of {column}")
        plt.grid(True)
        plt.show()

    def visualize_distributions(self, df, coin_names=None, columns_to_plot=5):
        if 'Date' in df.columns:
            df = df.set_index('Date')
        else:
            df = df.copy()
        
        if coin_names:
            columns = [coin for coin in coin_names if coin in df.columns]
        else:
            columns = df.columns

        num_columns = 4
        num_rows = (len(columns) + num_columns - 1) // num_columns

        plt.figure(figsize=(10, num_rows * 2))
        for i, column in enumerate(columns):
            plt.subplot(num_rows, num_columns, i + 1)
            plt.hist(df[column], bins=50, alpha=0.75, color='blue')
            plt.title(f'Distribution of {column}')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()