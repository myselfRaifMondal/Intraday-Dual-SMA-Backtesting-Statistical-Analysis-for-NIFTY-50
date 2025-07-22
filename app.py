import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import scipy.stats as stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from scipy.stats import skew, kurtosis, shapiro,jarque_bera
warnings.filterwarnings('ignore')

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class NiftyBacktester:
    """
    Comprehensive SMA Backtesting Class for NIFTY 50 Minute Data
    """

    def __init__(self, data_file='NIFTY 50_minute_data.csv', transaction_cost=0.00015):
        """
        Initialize the backtester

        Parameters:
        data_file: str, path to the CSV file
        transaction_cost: float, transaction cost per trade (default 0.015%)
        """
        self.data_file = data_file
        self.transaction_cost = transaction_cost
        self.sma_periods = [5, 10, 20, 50, 100, 200]
        self.results = {}
        self.best_sma = None

    def load_and_prepare_data(self):
        """Load and prepare the NIFTY 50 minute data"""
        print("Loading NIFTY 50 minute data...")

        df = pd.read_csv("NIFTY 50_minute_data.csv", parse_dates=True)
        df['DateTime'] = pd.to_datetime(df['DateTime'], format='%d-%m-%Y %H.%M')
        df.set_index('DateTime', inplace=True)
        self.df = df
        print(f"Data loaded: {len(self.df)} records from {self.df.index[0]} to {self.df.index[-1]}")
        print(f"Data shape: {self.df.shape}")
        print(f"\nData types:")
        print(self.df.dtypes)
        print(f"\nData info:")
        print(self.df.describe())

        return self.df

    def exploratory_data_analysis(self):
        """Comprehensive EDA of the NIFTY 50 data"""
        print("\n" + "=" * 60)
        print("EXPLORATORY DATA ANALYSIS")
        print("=" * 60)

        # Basic statistics
        print("\n1. BASIC STATISTICS")
        print("-" * 30)
        print(self.df.describe())

        # Check for missing values
        print(f"\n2. MISSING VALUES")
        print("-" * 30)
        print(self.df.isnull().sum())

        # Add time-based features
        self.df['hour'] = self.df.index.hour
        self.df['minute'] = self.df.index.minute
        self.df['day_of_week'] = self.df.index.dayofweek
        self.df['time_of_day'] = (self.df.index.hour - 9) * 60 + self.df.index.minute - 15

        # Calculate returns
        self.df['returns'] = self.df['Close'].pct_change()
        self.df['log_returns'] = np.log(self.df['Close'] / self.df['Close'].shift(1))

        # Volatility measures
        self.df['volatility_5'] = self.df['returns'].rolling(5).std()
        self.df['volatility_20'] = self.df['returns'].rolling(20).std()

        # Price features
        self.df['high_low_ratio'] = self.df['High'] / self.df['Low']
        self.df['open_close_ratio'] = self.df['Open'] / self.df['Close']

        print(f"\n3. RETURN STATISTICS")
        print("-" * 30)
        print(f"Mean Return: {self.df['returns'].mean():.6f}")
        print(f"Std Return: {self.df['returns'].std():.6f}")
        print(f"Skewness: {stats.skew(self.df['returns'].dropna()):.4f}")
        print(f"Kurtosis: {stats.kurtosis(self.df['returns'].dropna()):.4f}")

        # Test for normality
        shapiro_stat, shapiro_p = stats.shapiro(self.df['returns'].dropna().sample(5000))  # Sample for performance
        print(f"Shapiro-Wilk p-value: {shapiro_p:.2e}")
        print(f"Returns are {'NOT ' if shapiro_p < 0.05 else ''}normally distributed")

        # Create visualizations
        self._create_eda_plots()

        return self.df

    def _create_eda_plots(self):
        """Create comprehensive EDA visualizations"""

        # Set up the plotting environment
        fig = plt.figure(figsize=(20, 24))

        # 1. Price and Volume Time Series
        plt.subplot(4, 3, 1)
        plt.plot(self.df.index, self.df['Close'], linewidth=0.8, color='navy')
        plt.title('NIFTY 50 Price Evolution', fontsize=14, fontweight='bold')
        plt.ylabel('Price (INR)')
        plt.grid(True, alpha=0.3)

        plt.subplot(4, 3, 2)
        plt.plot(self.df.index, self.df['volume'], linewidth=0.5, color='orange')
        plt.title('Volume Evolution', fontsize=14, fontweight='bold')
        plt.ylabel('Volume')
        plt.grid(True, alpha=0.3)

        # 2. Return Distribution
        plt.subplot(4, 3, 3)
        returns_clean = self.df['returns'].dropna()
        plt.hist(returns_clean, bins=100, alpha=0.7, color='green', density=True)
        plt.axvline(returns_clean.mean(), color='red', linestyle='--', label=f'Mean: {returns_clean.mean():.6f}')
        plt.axvline(returns_clean.median(), color='orange', linestyle='--',
                    label=f'Median: {returns_clean.median():.6f}')
        plt.title('Return Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Returns')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 3. QQ Plot for Returns
        plt.subplot(4, 3, 4)
        stats.probplot(returns_clean.sample(5000), dist="norm", plot=plt)
        plt.title('Q-Q Plot: Returns vs Normal', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

        # 5. Intraday Patterns - Volatility
        plt.subplot(4, 3, 6)
        hourly_vol = self.df.groupby('hour')['returns'].std()
        plt.plot(hourly_vol.index, hourly_vol.values, marker='o', linewidth=2, markersize=6, color='red')
        plt.title('Hourly Volatility Pattern', fontsize=14, fontweight='bold')
        plt.xlabel('Hour of Day')
        plt.ylabel('Return Volatility')
        plt.grid(True, alpha=0.3)

        # 6. Day of Week Effect
        plt.subplot(4, 3, 7)
        daily_returns = self.df.groupby('day_of_week')['returns'].mean()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        plt.bar(range(len(days)), daily_returns.values, color='purple', alpha=0.7)
        plt.title('Day of Week Return Pattern', fontsize=14, fontweight='bold')
        plt.xlabel('Day of Week')
        plt.ylabel('Average Return')
        plt.xticks(range(len(days)), days)
        plt.grid(True, alpha=0.3)

        # 7. Autocorrelation of Returns
        plt.subplot(4, 3, 8)
        from statsmodels.tsa.stattools import acf
        lags = 50
        autocorr = acf(returns_clean, nlags=lags, fft=True)
        plt.plot(range(lags + 1), autocorr, color='darkgreen')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.axhline(y=1.96 / np.sqrt(len(returns_clean)), color='red', linestyle='--', alpha=0.7)
        plt.axhline(y=-1.96 / np.sqrt(len(returns_clean)), color='red', linestyle='--', alpha=0.7)
        plt.title('Autocorrelation of Returns', fontsize=14, fontweight='bold')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.grid(True, alpha=0.3)

        # 8. High-Low Range Analysis
        plt.subplot(4, 3, 9)
        range_pct = (self.df['High'] - self.df['Low']) / self.df['Close'] * 100
        plt.hist(range_pct, bins=50, alpha=0.7, color='orange')
        plt.title('Intraday Range Distribution (%)', fontsize=14, fontweight='bold')
        plt.xlabel('High-Low Range (%)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)

        # 9. Volume-Return Relationship
        plt.subplot(4, 3, 10)
        sample_data = self.df.dropna().sample(10000)  # Sample for performance
        plt.scatter(sample_data['volume'], sample_data['returns'], alpha=0.5, s=1)
        plt.title('Volume vs Returns Relationship', fontsize=14, fontweight='bold')
        plt.xlabel('Volume')
        plt.ylabel('Returns')
        plt.grid(True, alpha=0.3)

        # 10. Rolling Volatility
        plt.subplot(4, 3, 11)
        rolling_vol = self.df['returns'].rolling(1000).std()
        plt.plot(self.df.index, rolling_vol, color='red', linewidth=0.8)
        plt.title('Rolling Volatility (1000-period)', fontsize=14, fontweight='bold')
        plt.ylabel('Volatility')
        plt.grid(True, alpha=0.3)

        # 11. Price vs SMA Preview
        plt.subplot(4, 3, 12)
        sma_20 = self.df['Close'].rolling(20).mean()
        sample_idx = slice(-2000, None)  # Last 2000 points
        plt.plot(self.df.index[sample_idx], self.df['Close'][sample_idx], label='Close', linewidth=0.8)
        plt.plot(self.df.index[sample_idx], sma_20[sample_idx], label='SMA(20)', linewidth=1.2)
        plt.title('Price vs SMA(20) - Recent Data', fontsize=14, fontweight='bold')
        plt.ylabel('Price (INR)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Additional statistical tests
        print(f"\n4. ADDITIONAL STATISTICAL TESTS")
        print("-" * 30)

        # ARCH test for heteroscedasticity
        try:
            from statsmodels.stats.diagnostic import het_arch
            arch_stat, arch_pvalue, _, _ = het_arch(returns_clean.dropna(), nlags=5)
            print(f"ARCH test p-value: {arch_pvalue:.4f}")
            print(
                f"Returns {'exhibit' if arch_pvalue < 0.05 else 'do not exhibit'} ARCH effects (volatility clustering)")
        except:
            print("ARCH test could not be performed")

    def feature_engineering(self):
        """Create features for SMA backtesting"""
        print(f"\n5. FEATURE ENGINEERING")
        print("-" * 30)

        # Calculate all SMA periods
        for period in self.sma_periods:
            self.df[f'SMA_{period}'] = self.df['Close'].rolling(period).mean()
            self.df[f'price_above_sma_{period}'] = (self.df['Close'] > self.df[f'SMA_{period}']).astype(int)
            self.df[f'sma_slope_{period}'] = self.df[f'SMA_{period}'].diff()

        # Distance from SMA
        for period in self.sma_periods:
            self.df[f'dist_from_sma_{period}'] = (self.df['Close'] - self.df[f'SMA_{period}']) / self.df[
                f'SMA_{period}']

        # SMA crossovers
        for i, period in enumerate(self.sma_periods[:-1]):
            fast_sma = f'SMA_{period}'
            slow_sma = f'SMA_{self.sma_periods[i + 1]}'
            self.df[f'{fast_sma}_above_{slow_sma}'] = (self.df[fast_sma] > self.df[slow_sma]).astype(int)

        # Technical indicators
        self.df['rsi'] = self._calculate_rsi(self.df['Close'], 14)
        self.df['bb_upper'], self.df['bb_lower'] = self._calculate_bollinger_bands(self.df['Close'], 20)

        # Momentum indicators
        self.df['momentum_5'] = self.df['Close'] / self.df['Close'].shift(5) - 1
        self.df['momentum_20'] = self.df['Close'] / self.df['Close'].shift(20) - 1

        # Volume indicators
        self.df['volume_sma_20'] = self.df['volume'].rolling(20).mean()
        self.df['volume_ratio'] = self.df['volume'] / self.df['volume_sma_20']

        print(f" Features created for {len(self.sma_periods)} SMA periods")
        print(f"Total features: {self.df.shape[1]}")

        return self.df

    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band

    def backtest_single_sma(self, sma_period):
        """Backtest a single SMA strategy"""
        # Create signals
        if 'returns' not in self.df.columns:
            # simple pct-change returns
            self.df['returns'] = self.df['Close'].pct_change()
        sma_col = f'SMA_{sma_period}'
        signal_col = f'signal_{sma_period}'
        # Long when price > SMA, flat otherwise
        self.df[signal_col] = np.where(self.df['Close'] > self.df[sma_col], 1, 0)
        # Calculate position changes (entries and exits)
        self.df[f'position_change_{sma_period}'] = self.df[signal_col].diff()
        # Calculate strategy returns
        self.df[f'strategy_returns_{sma_period}'] = self.df[signal_col].shift(1) * self.df['returns']
        # Account for transaction costs
        transaction_costs = abs(self.df[f'position_change_{sma_period}']) * self.transaction_cost
        self.df[f'net_returns_{sma_period}'] = self.df[f'strategy_returns_{sma_period}'] - transaction_costs
        # Calculate cumulative returns
        self.df[f'cum_returns_{sma_period}'] = (1 + self.df[f'net_returns_{sma_period}']).cumprod()
        # Calculate metrics
        metrics = self._calculate_performance_metrics(sma_period)
        return metrics

    def _calculate_performance_metrics(self, sma_period):
        """Calculate comprehensive performance metrics for a strategy"""
        returns_col = f'net_returns_{sma_period}'
        cum_returns_col = f'cum_returns_{sma_period}'
        signal_col = f'signal_{sma_period}'
        # Clean data
        strategy_returns = self.df[returns_col].dropna()
        cum_returns = self.df[cum_returns_col].dropna()
        signals = self.df[signal_col].dropna()
        # Total return
        total_return = cum_returns.iloc[-1] - 1
        # Annualized return (assuming 250 trading days, 375 minutes per day)
        n_periods = len(strategy_returns)
        n_years = n_periods / (250 * 375)
        annualized_return = (1 + total_return) ** (1 / n_years) - 1
        # Volatility (annualized)
        volatility = strategy_returns.std() * np.sqrt(250 * 375)
        # Sharpe ratio
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        # Maximum drawdown
        rolling_max = cum_returns.expanding().max()
        drawdown = (cum_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        # Win rate
        winning_trades = strategy_returns[strategy_returns > 0]
        total_trades = len(strategy_returns[strategy_returns != 0])
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        # Trade statistics
        position_changes = self.df[f'position_change_{sma_period}'].abs().sum()
        n_trades = int(position_changes / 2)  # Round trips
        # Profit factor
        gross_profit = winning_trades.sum()
        gross_loss = abs(strategy_returns[strategy_returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        # Average trade
        avg_trade = strategy_returns[strategy_returns != 0].mean()
        # Best and worst trades
        best_trade = strategy_returns.max()
        worst_trade = strategy_returns.min()
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else np.inf
        # Sortino ratio
        downside_returns = strategy_returns[strategy_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(250 * 375)
        sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else np.inf

        # Time in market
        time_in_market = signals.mean()

        metrics = {
            'SMA_Period': sma_period,
            'Total_Return': total_return,
            'Annualized_Return': annualized_return,
            'Volatility': volatility,
            'Sharpe_Ratio': sharpe_ratio,
            'Max_Drawdown': max_drawdown,
            'Win_Rate': win_rate,
            'Profit_Factor': profit_factor,
            'Calmar_Ratio': calmar_ratio,
            'Sortino_Ratio': sortino_ratio,
            'Num_Trades': n_trades,
            'Avg_Trade': avg_trade,
            'Best_Trade': best_trade,
            'Worst_Trade': worst_trade,
            'Time_in_Market': time_in_market
        }

        return metrics

    def run_backtest(self):
        """Run backtest for all SMA periods"""
        print(f"\n6. RUNNING SMA BACKTESTS")
        print("-" * 30)
        all_metrics = []
        for sma_period in self.sma_periods:
            print(f"Backtesting SMA({sma_period})...")
            metrics = self.backtest_single_sma(sma_period)
            all_metrics.append(metrics)
            # Store individual results
            self.results[sma_period] = metrics
        # Create results DataFrame
        self.results_df = pd.DataFrame(all_metrics)
        # ind best performing SMA
        self.best_sma = self.results_df.loc[self.results_df['Sharpe_Ratio'].idxmax(), 'SMA_Period']
        print(f"Backtest complete!")
        print(f"Best performing SMA: {self.best_sma} (Sharpe: {self.results_df['Sharpe_Ratio'].max():.3f})")

        return self.results_df

    def create_performance_visualizations(self):
        """Create comprehensive performance visualizations"""
        print(f"\n7. CREATING PERFORMANCE VISUALIZATIONS")
        print("-" * 30)

        # Main performance dashboard
        fig = plt.figure(figsize=(20, 16))

        # 1. Equity Curves
        plt.subplot(3, 3, 1)
        for sma_period in self.sma_periods:
            cum_returns_col = f'cum_returns_{sma_period}'
            if cum_returns_col in self.df.columns:
                plt.plot(self.df.index, self.df[cum_returns_col], label=f'SMA({sma_period})', linewidth=1.5)

        # Add buy & hold
        buy_hold_cum = (1 + self.df['returns']).cumprod()
        plt.plot(self.df.index, buy_hold_cum, label='Buy & Hold', linewidth=2, linestyle='--', color='black')

        plt.title('Cumulative Returns - All SMA Strategies', fontsize=14, fontweight='bold')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')

        # 2. Drawdown Analysis
        plt.subplot(3, 3, 2)
        best_cum_col = f'cum_returns_{self.best_sma}'
        if best_cum_col in self.df.columns:
            cum_returns = self.df[best_cum_col].dropna()
            rolling_max = cum_returns.expanding().max()
            drawdown = (cum_returns - rolling_max) / rolling_max
            plt.fill_between(self.df.index[:len(drawdown)], drawdown, 0, alpha=0.7, color='red')
            plt.title(f'Drawdown - Best SMA({self.best_sma})', fontsize=14, fontweight='bold')
            plt.ylabel('Drawdown (%)')
            plt.grid(True, alpha=0.3)

        # 3. Performance Metrics Bar Chart
        plt.subplot(3, 3, 3)
        metrics_to_plot = ['Sharpe_Ratio', 'Calmar_Ratio', 'Win_Rate']
        x = np.arange(len(self.sma_periods))
        width = 0.25

        for i, metric in enumerate(metrics_to_plot):
            values = self.results_df[metric].values
            plt.bar(x + i * width, values, width, label=metric, alpha=0.8)

        plt.xlabel('SMA Period')
        plt.ylabel('Metric Value')
        plt.title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
        plt.xticks(x + width, self.sma_periods)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 4. Return Distribution - Best SMA
        plt.subplot(3, 3, 4)
        best_returns_col = f'net_returns_{self.best_sma}'
        if best_returns_col in self.df.columns:
            returns = self.df[best_returns_col].dropna()
            returns = returns[returns != 0]  # Only trading days
            plt.hist(returns, bins=50, alpha=0.7, density=True, color='green')
            plt.axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.6f}')
            plt.title(f'Return Distribution - SMA({self.best_sma})', fontsize=14, fontweight='bold')
            plt.xlabel('Returns')
            plt.ylabel('Density')
            plt.legend()
            plt.grid(True, alpha=0.3)

        # 5. Rolling Sharpe Ratio
        plt.subplot(3, 3, 5)
        window = 1000  # 1000-minute rolling window
        for sma_period in [self.best_sma, 20, 50]:  # Show best + couple others
            returns_col = f'net_returns_{sma_period}'
            if returns_col in self.df.columns:
                returns = self.df[returns_col]
                rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(250 * 375)
                plt.plot(self.df.index, rolling_sharpe, label=f'SMA({sma_period})', linewidth=1.2)

        plt.title('Rolling Sharpe Ratio (1000-period)', fontsize=14, fontweight='bold')
        plt.ylabel('Sharpe Ratio')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 6. Trade Analysis
        plt.subplot(3, 3, 6)
        trade_metrics = ['Num_Trades', 'Win_Rate', 'Profit_Factor']
        sma_labels = [f'SMA({p})' for p in self.sma_periods]

        # Normalize metrics for comparison
        normalized_data = self.results_df[trade_metrics].copy()
        for col in trade_metrics:
            normalized_data[col] = normalized_data[col] / normalized_data[col].max()

        x = np.arange(len(self.sma_periods))
        width = 0.25

        for i, metric in enumerate(trade_metrics):
            plt.bar(x + i * width, normalized_data[metric], width, label=metric, alpha=0.8)

        plt.xlabel('SMA Period')
        plt.ylabel('Normalized Value')
        plt.title('Trade Metrics (Normalized)', fontsize=14, fontweight='bold')
        plt.xticks(x + width, self.sma_periods)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 7. Risk-Return Scatter
        plt.subplot(3, 3, 7)
        plt.scatter(self.results_df['Volatility'], self.results_df['Annualized_Return'],
                    s=100, alpha=0.7, c=self.results_df['Sharpe_Ratio'], cmap='viridis')

        for i, sma in enumerate(self.sma_periods):
            plt.annotate(f'SMA({sma})',
                         (self.results_df.iloc[i]['Volatility'], self.results_df.iloc[i]['Annualized_Return']),
                         xytext=(5, 5), textcoords='offset points', fontsize=9)

        plt.xlabel('Volatility (Annualized)')
        plt.ylabel('Return (Annualized)')
        plt.title('Risk-Return Profile', fontsize=14, fontweight='bold')
        plt.colorbar(label='Sharpe Ratio')
        plt.grid(True, alpha=0.3)

        # 8. Maximum Drawdown Comparison
        plt.subplot(3, 3, 8)
        mdd_values = [abs(x) for x in self.results_df['Max_Drawdown'].values]
        colors = ['red' if x == max(mdd_values) else 'lightblue' for x in mdd_values]
        bars = plt.bar(range(len(self.sma_periods)), mdd_values, color=colors, alpha=0.7)
        plt.xlabel('SMA Period')
        plt.ylabel('Maximum Drawdown (%)')
        plt.title('Maximum Drawdown Comparison', fontsize=14, fontweight='bold')
        plt.xticks(range(len(self.sma_periods)), [f'SMA({p})' for p in self.sma_periods], rotation=45)
        plt.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, mdd_values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                     f'{value:.3f}', ha='center', va='bottom', fontsize=10)

        # 9. Strategy Heat Map
        plt.subplot(3, 3, 9)
        heatmap_data = self.results_df[['SMA_Period', 'Sharpe_Ratio', 'Max_Drawdown', 'Win_Rate']].set_index(
            'SMA_Period').T

        # Normalize for better visualization
        heatmap_normalized = heatmap_data.copy()
        heatmap_normalized.loc['Max_Drawdown'] = abs(heatmap_normalized.loc['Max_Drawdown'])  # Make positive
        for row in heatmap_normalized.index:
            heatmap_normalized.loc[row] = heatmap_normalized.loc[row] / heatmap_normalized.loc[row].max()

        sns.heatmap(heatmap_normalized, annot=True, cmap='RdYlGn', center=0.5, fmt='.3f')
        plt.title('Performance Heat Map (Normalized)', fontsize=14, fontweight='bold')
        plt.ylabel('Metrics')

        plt.tight_layout()
        plt.savefig('performance_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Create additional detailed charts
        self._create_detailed_analysis_charts()

    def _create_detailed_analysis_charts(self):
        """Create detailed analysis charts for the best performing SMA"""
        best_sma = self.best_sma
        # Detailed analysis of best SMA
        fig = plt.figure(figsize=(20, 12))
        # 1. Price vs SMA with signals
        plt.subplot(2, 3, 1)
        sample_data = self.df.iloc[-5000:]  # Last 5000 points for clarity
        plt.plot(sample_data.index, sample_data['Close'], label='NIFTY 50', linewidth=1, alpha=0.8)
        plt.plot(sample_data.index, sample_data[f'SMA_{best_sma}'], label=f'SMA({best_sma})', linewidth=2)
        # Highlight long positions
        long_positions = sample_data[sample_data[f'signal_{best_sma}'] == 1]
        plt.scatter(long_positions.index, long_positions['Close'], color='green', s=1, alpha=0.5, label='Long Position')
        plt.title(f'NIFTY 50 vs SMA({best_sma}) - Recent Data', fontsize=14, fontweight='bold')
        plt.ylabel('Price (INR)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. Monthly Returns Heat Map
        plt.subplot(2, 3, 2)
        returns_col = f'net_returns_{best_sma}'
        if returns_col in self.df.columns:
            monthly_returns = self.df[returns_col].resample('M').sum()
            monthly_returns_df = pd.DataFrame({
                'Year': monthly_returns.index.year,
                'Month': monthly_returns.index.month,
                'Return': monthly_returns.values
            })

            pivot_table = monthly_returns_df.pivot(index='Month', columns='Year', values='Return')
            sns.heatmap(pivot_table, annot=True, cmap='RdYlGn', center=0, fmt='.3f')
            plt.title(f'Monthly Returns Heat Map - SMA({best_sma})', fontsize=14, fontweight='bold')
            plt.ylabel('Month')
        # 3. Trade Duration Analysis
        plt.subplot(2, 3, 3)
        signal_col = f'signal_{best_sma}'
        if signal_col in self.df.columns:
            # Calculate trade durations
            signals = self.df[signal_col].copy()
            trade_durations = []
            current_duration = 0

            for signal in signals:
                if signal == 1:
                    current_duration += 1
                else:
                    if current_duration > 0:
                        trade_durations.append(current_duration)
                        current_duration = 0

            if trade_durations:
                plt.hist(trade_durations, bins=30, alpha=0.7, color='blue')
                plt.axvline(np.mean(trade_durations), color='red', linestyle='--',
                            label=f'Mean: {np.mean(trade_durations):.1f} minutes')
                plt.title('Trade Duration Distribution', fontsize=14, fontweight='bold')
                plt.xlabel('Duration (minutes)')
                plt.ylabel('Frequency')
                plt.legend()
                plt.grid(True, alpha=0.3)
        # 4. Underwater Curve
        plt.subplot(2, 3, 4)
        cum_returns_col = f'cum_returns_{best_sma}'
        if cum_returns_col in self.df.columns:
            cum_returns = self.df[cum_returns_col].dropna()
            rolling_max = cum_returns.expanding().max()
            underwater = (cum_returns - rolling_max) / rolling_max

            plt.fill_between(self.df.index[:len(underwater)], underwater, 0, alpha=0.7, color='red')
            plt.title(f'Underwater Curve - SMA({best_sma})', fontsize=14, fontweight='bold')
            plt.ylabel('Drawdown from Peak')
            plt.grid(True, alpha=0.3)
        # 5. Return vs Benchmark
        plt.subplot(2, 3, 5)
        benchmark_cum = (1 + self.df['returns']).cumprod()
        strategy_cum = self.df[f'cum_returns_{best_sma}']
        plt.plot(self.df.index, benchmark_cum, label='Buy & Hold', linewidth=2)
        plt.plot(self.df.index, strategy_cum, label=f'SMA({best_sma})', linewidth=2)
        plt.title('Strategy vs Benchmark', fontsize=14, fontweight='bold')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        # 6. Rolling Metrics
        plt.subplot(2, 3, 6)
        window = 2000
        returns = self.df[f'net_returns_{best_sma}'].dropna()
        rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(250 * 375)
        rolling_return = returns.rolling(window).mean() * 250 * 375

        ax1 = plt.gca()
        ax2 = ax1.twinx()

        line1 = ax1.plot(self.df.index[:len(rolling_sharpe)], rolling_sharpe, 'b-', label='Rolling Sharpe')
        line2 = ax2.plot(self.df.index[:len(rolling_return)], rolling_return, 'r-', label='Rolling Return')

        ax1.set_xlabel('Time')
        ax1.set_ylabel('Sharpe Ratio', color='b')
        ax2.set_ylabel('Annualized Return', color='r')

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')

        plt.title(f'Rolling Performance Metrics - SMA({best_sma})', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('detailed_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def statistical_analysis_best_sma(self):
        """Perform detailed statistical analysis of the best SMA strategy"""
        print(f"\n8. STATISTICAL ANALYSIS - BEST SMA({self.best_sma})")
        print("-" * 50)

        returns_col = f'net_returns_{self.best_sma}'
        returns = self.df[returns_col].dropna()
        returns_trading = returns[returns != 0]  # Only trading periods

        # Basic statistics
        print("RETURN STATISTICS:")
        print(f"Mean return: {returns_trading.mean():.6f} ({returns_trading.mean() * 250 * 375 * 100:.2f}% annualized)")
        print(f"Median return: {returns_trading.median():.6f}")
        print(
            f"Standard deviation: {returns_trading.std():.6f} ({returns_trading.std() * np.sqrt(250 * 375) * 100:.2f}% annualized)")
        print(f"Skewness: {stats.skew(returns_trading):.4f}")
        print(f"Kurtosis: {stats.kurtosis(returns_trading):.4f}")
        print(f"Minimum return: {returns_trading.min():.6f} ({returns_trading.min() * 100:.3f}%)")
        print(f"Maximum return: {returns_trading.max():.6f} ({returns_trading.max() * 100:.3f}%)")

        # Percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        print(f"\nPERCENTILES:")
        for p in percentiles:
            value = np.percentile(returns_trading, p)
            print(f"{p:2d}th percentile: {value:.6f} ({value * 100:.3f}%)")

        # Statistical tests
        print(f"\nSTATISTICAL TESTS:")

        # Normality test
        if len(returns_trading) > 5000:
            sample_returns = returns_trading.sample(5000)
        else:
            sample_returns = returns_trading

        shapiro_stat, shapiro_p = stats.shapiro(sample_returns)
        print(f"Shapiro-Wilk test: statistic={shapiro_stat:.4f}, p-value={shapiro_p:.2e}")
        print(f"Returns are {'NOT ' if shapiro_p < 0.05 else ''}normally distributed")

        # Jarque-Bera test
        jb_stat, jb_p = stats.jarque_bera(returns_trading)
        print(f"Jarque-Bera test: statistic={jb_stat:.4f}, p-value={jb_p:.2e}")

        # One-sample t-test (test if mean is significantly different from 0)
        t_stat, t_p = stats.ttest_1samp(returns_trading, 0)
        print(f"T-test (mean=0): statistic={t_stat:.4f}, p-value={t_p:.2e}")
        print(f"Mean return is {'significantly' if t_p < 0.05 else 'not significantly'} different from 0")

        # Create distribution analysis plots
        self._create_distribution_plots(returns_trading)
        # Win/Loss analysis
        self._analyze_win_loss_patterns(returns_trading)
        # Regime analysis
        self._analyze_market_regimes()

    def _create_distribution_plots(self, returns):
        """Create detailed distribution analysis plots"""
        fig = plt.figure(figsize=(16, 12))
        # 1. Return distribution with normal overlay
        plt.subplot(2, 3, 1)
        plt.hist(returns, bins=100, density=True, alpha=0.7, color='skyblue', label='Actual')
        # Overlay normal distribution
        mu, sigma = returns.mean(), returns.std()
        x = np.linspace(returns.min(), returns.max(), 100)
        normal_curve = stats.norm.pdf(x, mu, sigma)
        plt.plot(x, normal_curve, 'r-', linewidth=2, label='Normal')
        plt.title('Return Distribution vs Normal', fontsize=14, fontweight='bold')
        plt.xlabel('Returns')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        # 2. Log-scale distribution
        plt.subplot(2, 3, 2)
        plt.hist(returns, bins=100, density=True, alpha=0.7, color='green')
        plt.yscale('log')
        plt.title('Return Distribution (Log Scale)', fontsize=14, fontweight='bold')
        plt.xlabel('Returns')
        plt.ylabel('Log Density')
        plt.grid(True, alpha=0.3)
        # 3. Q-Q plot
        plt.subplot(2, 3, 3)
        stats.probplot(returns, dist="norm", plot=plt)
        plt.title('Q-Q Plot vs Normal Distribution', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        # 4. Cumulative distribution
        plt.subplot(2, 3, 4)
        sorted_returns = np.sort(returns)
        cumulative_prob = np.arange(1, len(sorted_returns) + 1) / len(sorted_returns)
        plt.plot(sorted_returns, cumulative_prob, linewidth=2)
        plt.title('Cumulative Distribution Function', fontsize=14, fontweight='bold')
        plt.xlabel('Returns')
        plt.ylabel('Cumulative Probability')
        plt.grid(True, alpha=0.3)
        # 5. Box plot with outliers
        plt.subplot(2, 3, 5)
        box_data = [returns[returns > 0], returns[returns < 0], returns]
        plt.boxplot(box_data, labels=['Winning', 'Losing', 'All'])
        plt.title('Return Distribution Box Plot', fontsize=14, fontweight='bold')
        plt.ylabel('Returns')
        plt.grid(True, alpha=0.3)
        # 6. Tail analysis
        plt.subplot(2, 3, 6)
        # Plot tail distributions
        tail_threshold = 0.05  # 5%
        left_tail = returns[returns <= np.percentile(returns, tail_threshold * 100)]
        right_tail = returns[returns >= np.percentile(returns, (1 - tail_threshold) * 100)]
        plt.hist(left_tail, bins=20, alpha=0.7, color='red', label=f'Left tail ({tail_threshold * 100}%)')
        plt.hist(right_tail, bins=20, alpha=0.7, color='green', label=f'Right tail ({tail_threshold * 100}%)')
        plt.title('Tail Distribution Analysis', fontsize=14, fontweight='bold')
        plt.xlabel('Returns')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        # Print tail statistics
        print(f"\nTAIL ANALYSIS:")
        print(f"Left tail (bottom 5%) statistics:")
        print(f"  Mean: {left_tail.mean():.6f}")
        print(f"  Std: {left_tail.std():.6f}")
        print(f"  Count: {len(left_tail)}")
        print(f"  Contribution to total return: {left_tail.sum():.6f}")
        print(f"Right tail (top 5%) statistics:")
        print(f"  Mean: {right_tail.mean():.6f}")
        print(f"  Std: {right_tail.std():.6f}")
        print(f"  Count: {len(right_tail)}")
        print(f"  Contribution to total return: {right_tail.sum():.6f}")

    def _analyze_win_loss_patterns(self, returns):
        """Analyze win/loss patterns and streaks"""

        print(f"\nWIN/LOSS PATTERN ANALYSIS:")
        print("-" * 30)

        # Separate wins and losses
        wins = returns[returns > 0]
        losses = returns[returns < 0]

        print(f"Winning trades: {len(wins)} ({len(wins) / len(returns) * 100:.1f}%)")
        print(f"Losing trades: {len(losses)} ({len(losses) / len(returns) * 100:.1f}%)")
        print(f"Neutral trades: {len(returns) - len(wins) - len(losses)}")

        print(f"\nWINNING TRADES:")
        print(f"  Average: {wins.mean():.6f} ({wins.mean() * 100:.3f}%)")
        print(f"  Median: {wins.median():.6f}")
        print(f"  Best: {wins.max():.6f} ({wins.max() * 100:.3f}%)")
        print(f"  Total contribution: {wins.sum():.6f}")

        print(f"\nLOSING TRADES:")
        print(f"  Average: {losses.mean():.6f} ({losses.mean() * 100:.3f}%)")
        print(f"  Median: {losses.median():.6f}")
        print(f"  Worst: {losses.min():.6f} ({losses.min() * 100:.3f}%)")
        print(f"  Total contribution: {losses.sum():.6f}")

        # Streak analysis
        win_signals = (returns > 0).astype(int)
        streaks = []
        current_streak = 0
        streak_type = None

        for signal in win_signals:
            if signal == 1:  # Win
                if streak_type == 'win':
                    current_streak += 1
                else:
                    if current_streak > 0:
                        streaks.append(('loss', current_streak))
                    current_streak = 1
                    streak_type = 'win'
            else:  # Loss
                if streak_type == 'loss':
                    current_streak += 1
                else:
                    if current_streak > 0:
                        streaks.append(('win', current_streak))
                    current_streak = 1
                    streak_type = 'loss'

        # Add final streak
        if current_streak > 0:
            streaks.append((streak_type, current_streak))

        win_streaks = [s[1] for s in streaks if s[0] == 'win']
        loss_streaks = [s[1] for s in streaks if s[0] == 'loss']

        print(f"\nSTREAK ANALYSIS:")
        if win_streaks:
            print(f"Win streaks - Max: {max(win_streaks)}, Avg: {np.mean(win_streaks):.1f}")
        if loss_streaks:
            print(f"Loss streaks - Max: {max(loss_streaks)}, Avg: {np.mean(loss_streaks):.1f}")

    def _analyze_market_regimes(self):
        """Analyze strategy performance in different market regimes"""

        print(f"\nMARKET REGIME ANALYSIS:")
        print("-" * 30)

        # Define regimes based on volatility
        volatility = self.df['returns'].rolling(100).std()
        vol_percentiles = volatility.quantile([0.33, 0.67])

        low_vol_mask = volatility <= vol_percentiles.iloc[0]
        medium_vol_mask = (volatility > vol_percentiles.iloc[0]) & (volatility <= vol_percentiles.iloc[1])
        high_vol_mask = volatility > vol_percentiles.iloc[1]

        returns_col = f'net_returns_{self.best_sma}'
        strategy_returns = self.df[returns_col]

        # Performance in different volatility regimes
        low_vol_returns = strategy_returns[low_vol_mask]
        medium_vol_returns = strategy_returns[medium_vol_mask]
        high_vol_returns = strategy_returns[high_vol_mask]

        print(f"LOW VOLATILITY REGIME:")
        print(f"  Mean return: {low_vol_returns.mean():.6f}")
        print(f"  Sharpe ratio: {low_vol_returns.mean() / low_vol_returns.std() * np.sqrt(250 * 375):.3f}")
        print(f"  Win rate: {(low_vol_returns > 0).mean():.3f}")

        print(f"\nMEDIUM VOLATILITY REGIME:")
        print(f"  Mean return: {medium_vol_returns.mean():.6f}")
        print(f"  Sharpe ratio: {medium_vol_returns.mean() / medium_vol_returns.std() * np.sqrt(250 * 375):.3f}")
        print(f"  Win rate: {(medium_vol_returns > 0).mean():.3f}")

        print(f"\nHIGH VOLATILITY REGIME:")
        print(f"  Mean return: {high_vol_returns.mean():.6f}")
        print(f"  Sharpe ratio: {high_vol_returns.mean() / high_vol_returns.std() * np.sqrt(250 * 375):.3f}")
        print(f"  Win rate: {(high_vol_returns > 0).mean():.3f}")

    #############################now lets do dual sma crossover###############################
    def analyze_strategy_performance(self, strategy_returns, position_changes, title_prefix=''):
        """Original single SMA performance analysis function (unchanged)"""
        cum_returns = (1 + strategy_returns).cumprod()
        total_return = cum_returns.iloc[-1] - 1
        volatility = strategy_returns.std() * np.sqrt(250 * 375)
        sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(250 * 375)
        rolling_max = cum_returns.cummax()
        drawdown = (cum_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Win rate
        trade_returns = strategy_returns[position_changes == 1].dropna()
        win_rate = (trade_returns > 0).mean() if len(trade_returns) > 0 else np.nan

        # Distribution stats
        mean_ret = trade_returns.mean() if not trade_returns.empty else np.nan
        median_ret = trade_returns.median() if not trade_returns.empty else np.nan
        std_ret = trade_returns.std() if not trade_returns.empty else np.nan
        skew_ret = skew(trade_returns) if len(trade_returns) > 0 else np.nan
        kurt_ret = kurtosis(trade_returns) if len(trade_returns) > 0 else np.nan
        shapiro_p = shapiro(trade_returns)[1] if len(trade_returns) >= 3 else np.nan

        print(f"\n{title_prefix} Strategy Performance Metrics")
        print("-" * 40)
        print(f"Total Return: {total_return * 100:.2f}%")
        print(f"Annualized Sharpe Ratio: {sharpe:.2f}")
        print(f"Max Drawdown: {max_drawdown * 100:.2f}%")
        print(f"Volatility (Annualized): {volatility * 100:.2f}%")
        print(f"Win Rate: {win_rate * 100:.2f}%")
        print(f"Number of Trades: {len(trade_returns)}")
        print(f"Trade Return Mean: {mean_ret * 100:.3f}%")
        print(f"Trade Return Median: {median_ret * 100:.3f}%")
        print(f"Trade Return Std Dev: {std_ret * 100:.3f}%")
        print(f"Trade Return Skewness: {skew_ret:.3f}")
        print(f"Trade Return Kurtosis: {kurt_ret:.3f}")
        print(f"Shapiro-Wilk p-value: {shapiro_p:.3f}")

        # Original plots
        plt.figure(figsize=(15, 10))
        plt.subplot(221)
        cum_returns.plot(title=f'{title_prefix} Strategy Equity Curve')
        plt.ylabel('Cumulative Returns')
        plt.grid(True, alpha=0.3)

        plt.subplot(222)
        plt.hist(trade_returns * 100, bins=50, alpha=0.7, density=True)
        plt.title(f'{title_prefix} Trade Returns Distribution')
        plt.xlabel('Trade Return (%)')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)

        plt.subplot(223)
        drawdown.plot(title=f'{title_prefix} Drawdown', color='red')
        plt.ylabel('Drawdown')
        plt.grid(True, alpha=0.3)

        plt.subplot(224)
        trade_returns.plot(kind='box', vert=False)
        plt.title(f'{title_prefix} Trade Returns Boxplot')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def comprehensive_dual_sma_crossover_analysis(self):
        """
        COMPREHENSIVE DUAL SMA CROSSOVER ANALYSIS """
        print("\n" + "=" * 80)
        print(" COMPREHENSIVE DUAL SMA CROSSOVER ANALYSIS")
        print("=" * 80)
        # Extended dual SMA combinations for thorough analysis
        fast_periods = [5, 10, 20, 50]
        slow_periods = [20, 50, 100, 200]
        dual_results = []
        self.dual_equity_curves = {}
        self.dual_trade_data = {}
        signal_data = {}
        total_combinations = sum(1 for f in fast_periods for s in slow_periods if f < s)
        print(f"Testing {total_combinations} dual SMA combinations...")
        # Test all dual SMA combinations
        for i, fast in enumerate(fast_periods):
            for j, slow in enumerate(slow_periods):
                if fast < slow:
                    combo_name = f"{fast}/{slow}"
                    print(f"  → Analyzing {combo_name} combination... ({len(dual_results) + 1}/{total_combinations})")

                    # Create dual SMA strategy signals
                    fast_sma = self.df[f'SMA_{fast}']
                    slow_sma = self.df[f'SMA_{slow}']

                    # Signal: Long when fast SMA > slow SMA
                    signal = (fast_sma > slow_sma).astype(int)

                    # Calculate returns with transaction costs
                    strategy_returns = signal.shift(1) * self.df['returns']
                    position_changes = signal.diff().abs()
                    transaction_costs = position_changes * self.transaction_cost
                    net_returns = strategy_returns - transaction_costs

                    # Store data for detailed analysis
                    self.dual_equity_curves[combo_name] = (1 + net_returns.dropna()).cumprod()
                    trade_returns = net_returns[position_changes == 1].dropna()
                    self.dual_trade_data[combo_name] = trade_returns
                    signal_data[combo_name] = signal

                    # Calculate comprehensive metrics
                    cum_returns = self.dual_equity_curves[combo_name]
                    if len(cum_returns) == 0:
                        continue
                    total_return = cum_returns.iloc[-1] - 1
                    volatility = net_returns.std() * np.sqrt(250 * 375)
                    sharpe = (net_returns.mean() / net_returns.std()) * np.sqrt(
                        250 * 375) if net_returns.std() > 0 else 0

                    # Drawdown analysis
                    rolling_max = cum_returns.expanding().max()
                    drawdown = (cum_returns - rolling_max) / rolling_max
                    max_drawdown = drawdown.min()
                    avg_drawdown = drawdown.mean()
                    # Trade analysis
                    win_rate = (trade_returns > 0).mean() if len(trade_returns) > 0 else 0
                    avg_win = trade_returns[trade_returns > 0].mean() if len(
                        trade_returns[trade_returns > 0]) > 0 else 0
                    avg_loss = trade_returns[trade_returns < 0].mean() if len(
                        trade_returns[trade_returns < 0]) > 0 else 0
                    profit_factor = -avg_win / avg_loss * win_rate / (
                                1 - win_rate) if avg_loss < 0 and win_rate < 1 else np.inf

                    # Risk-adjusted metrics
                    calmar_ratio = total_return / abs(max_drawdown) if max_drawdown < 0 else np.inf
                    downside_returns = net_returns[net_returns < 0]
                    sortino_ratio = net_returns.mean() / downside_returns.std() * np.sqrt(250 * 375) if len(
                        downside_returns) > 0 else np.inf
                    # Distribution analysis
                    if len(trade_returns) > 2:
                        trade_skew = skew(trade_returns)
                        trade_kurt = kurtosis(trade_returns)
                        _, shapiro_p = shapiro(trade_returns) if len(trade_returns) >= 3 else (np.nan, np.nan)
                        _, jb_p = jarque_bera(trade_returns) if len(trade_returns) >= 3 else (np.nan, np.nan)
                    else:
                        trade_skew = trade_kurt = shapiro_p = jb_p = np.nan
                    # Add to results
                    dual_results.append({
                        'Fast_SMA': fast,
                        'Slow_SMA': slow,
                        'Strategy': combo_name,
                        'Total_Return': total_return,
                        'Annualized_Return': (1 + total_return) ** (250 * 375 / len(net_returns.dropna())) - 1,
                        'Volatility': volatility,
                        'Sharpe_Ratio': sharpe,
                        'Calmar_Ratio': calmar_ratio,
                        'Sortino_Ratio': sortino_ratio,
                        'Max_Drawdown': max_drawdown,
                        'Avg_Drawdown': avg_drawdown,
                        'Win_Rate': win_rate,
                        'Trade_Count': len(trade_returns),
                        'Avg_Win': avg_win,
                        'Avg_Loss': avg_loss,
                        'Profit_Factor': profit_factor,
                        'Trade_Skew': trade_skew,
                        'Trade_Kurt': trade_kurt,
                        'Shapiro_P': shapiro_p,
                        'JB_P': jb_p
                    })
        # Create results dataframe and sort by Sharpe ratio
        self.dual_sma_results = pd.DataFrame(dual_results)
        self.dual_sma_results = self.dual_sma_results.sort_values('Sharpe_Ratio', ascending=False)
        self.best_dual_strategy = self.dual_sma_results.iloc[0]['Strategy']
        print(f"\n Analysis complete! Best strategy: {self.best_dual_strategy}")
        print(f"   Sharpe Ratio: {self.dual_sma_results.iloc[0]['Sharpe_Ratio']:.3f}")
        print(f"   Total Return: {self.dual_sma_results.iloc[0]['Total_Return'] * 100:.2f}%")
        return self.dual_sma_results

    def create_dual_sma_comprehensive_dashboard(self):
        """Create extensive visualization dashboard for dual SMA analysis"""

        print("\n Creating Comprehensive Dual SMA Dashboard...")

        # Get best strategy data
        best_strategy = self.best_dual_strategy
        best_trades = self.dual_trade_data[best_strategy]
        best_equity = self.dual_equity_curves[best_strategy]

        # Create master dashboard with multiple subplots
        fig = plt.figure(figsize=(25, 18))
        fig.suptitle(' DUAL SMA CROSSOVER - COMPREHENSIVE ANALYSIS DASHBOARD',
                     fontsize=20, fontweight='bold', y=0.98)

        # 1. Main Equity Curves Comparison (Large plot - spans 2 columns)
        ax1 = plt.subplot(4, 4, (1, 2))
        colors = plt.cm.tab20(np.linspace(0, 1, len(self.dual_equity_curves)))
        for i, (combo, curve) in enumerate(self.dual_equity_curves.items()):
            if combo == best_strategy:
                ax1.plot(curve.index, curve.values, label=f'{combo}  BEST',
                         linewidth=4, color='red', alpha=0.9)
            else:
                ax1.plot(curve.index, curve.values, label=combo,
                         linewidth=2, alpha=0.6, color=colors[i])
        ax1.set_title(' Equity Curves - All Dual SMA Combinations', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cumulative Returns')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # 2. Performance Heatmap - Sharpe Ratios
        ax2 = plt.subplot(4, 4, 3)
        pivot_sharpe = self.dual_sma_results.pivot(index='Fast_SMA', columns='Slow_SMA', values='Sharpe_Ratio')
        sns.heatmap(pivot_sharpe, annot=True, fmt='.2f', ax=ax2, cmap='RdYlGn',
                    cbar_kws={'label': 'Sharpe Ratio'})
        ax2.set_title(' Sharpe Ratio Heatmap', fontweight='bold')

        # 3. Performance Heatmap - Total Returns
        ax3 = plt.subplot(4, 4, 4)
        pivot_returns = self.dual_sma_results.pivot(index='Fast_SMA', columns='Slow_SMA', values='Total_Return')

        sns.heatmap(pivot_returns * 100, annot=True, fmt='.1f', ax=ax3, cmap='RdYlGn',
                    cbar_kws={'label': 'Total Return (%)'})
        ax3.set_title(' Total Return Heatmap (%)', fontweight='bold')

        # 4. Best Strategy Drawdown Analysis
        ax4 = plt.subplot(4, 4, 5)
        rolling_max = best_equity.expanding().max()
        drawdown = (best_equity - rolling_max) / rolling_max
        ax4.fill_between(drawdown.index, drawdown.values * 100, 0, alpha=0.3, color='red')
        ax4.plot(drawdown.index, drawdown.values * 100, color='darkred', linewidth=1.5)
        ax4.set_title(f' Drawdown Analysis - {best_strategy}', fontweight='bold')
        ax4.set_ylabel('Drawdown (%)')
        ax4.grid(True, alpha=0.3)

        # 5. Trade Returns Distribution - Best Strategy with detailed stats
        ax5 = plt.subplot(4, 4, 6)
        ax5.hist(best_trades * 100, bins=50, alpha=0.7, density=True,
                 color='skyblue', edgecolor='black')
        ax5.axvline(best_trades.mean() * 100, color='red', linestyle='--',
                    linewidth=2, label=f'Mean: {best_trades.mean() * 100:.3f}%')
        ax5.axvline(best_trades.median() * 100, color='orange', linestyle='--',
                    linewidth=2, label=f'Median: {best_trades.median() * 100:.3f}%')
        # Add percentile lines
        p95 = np.percentile(best_trades * 100, 95)
        p5 = np.percentile(best_trades * 100, 5)
        ax5.axvline(p95, color='green', linestyle=':', alpha=0.7, label=f'95th: {p95:.3f}%')
        ax5.axvline(p5, color='red', linestyle=':', alpha=0.7, label=f'5th: {p5:.3f}%')

        ax5.set_title(f' Trade Returns Distribution - {best_strategy}', fontweight='bold')
        ax5.set_xlabel('Trade Return (%)')
        ax5.set_ylabel('Density')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)

        # 6. Risk-Return Scatter Plot with enhanced annotations
        ax6 = plt.subplot(4, 4, 7)
        scatter = ax6.scatter(self.dual_sma_results['Volatility'] * 100,
                              self.dual_sma_results['Total_Return'] * 100,
                              c=self.dual_sma_results['Sharpe_Ratio'],
                              s=self.dual_sma_results['Trade_Count'] / 10,  # Size by trade count
                              alpha=0.8, cmap='RdYlGn', edgecolors='black')
        plt.colorbar(scatter, ax=ax6, label='Sharpe Ratio')

        # Annotate best strategy
        best_row = self.dual_sma_results.iloc[0]
        ax6.annotate(f' {best_row["Strategy"]}',
                     (best_row['Volatility'] * 100, best_row['Total_Return'] * 100),
                     xytext=(10, 10), textcoords='offset points', fontsize=10,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        ax6.set_xlabel('Volatility (%) - Bubble size = Trade Count')
        ax6.set_ylabel('Total Return (%)')
        ax6.set_title(' Risk-Return-Frequency Profile', fontweight='bold')
        ax6.grid(True, alpha=0.3)

        # 7. Performance Metrics Comparison - Top 5
        ax7 = plt.subplot(4, 4, 8)
        top_5 = self.dual_sma_results.head(5)
        metrics = ['Sharpe_Ratio', 'Calmar_Ratio', 'Win_Rate', 'Profit_Factor']

        x = np.arange(len(top_5))
        width = 0.2

        for i, metric in enumerate(metrics):
            values = top_5[metric].fillna(0)
            # Cap extreme values for better visualization
            if metric == 'Profit_Factor':
                values = np.minimum(values, 5)
            elif metric in ['Sharpe_Ratio', 'Calmar_Ratio']:
                values = np.minimum(values, 3)

            bars = ax7.bar(x + i * width, values, width, label=metric, alpha=0.8)

            # Add value labels on bars
            for j, bar in enumerate(bars):
                height = bar.get_height()
                if height > 0:
                    ax7.text(bar.get_x() + bar.get_width() / 2., height,
                             f'{top_5.iloc[j][metric]:.2f}',
                             ha='center', va='bottom', fontsize=8)

        ax7.set_xlabel('Strategy Rank')
        ax7.set_ylabel('Metric Value (Capped for Display)')
        ax7.set_title(' Top 5 Strategies - Key Metrics', fontweight='bold')
        ax7.set_xticks(x + width * 1.5)
        ax7.set_xticklabels([f"{int(row['Fast_SMA'])}/{int(row['Slow_SMA'])}" for _, row in top_5.iterrows()],
                            rotation=45)
        ax7.legend(fontsize=8)
        ax7.grid(True, alpha=0.3)

        # 8. Win/Loss Analysis with enhanced statistics
        ax8 = plt.subplot(4, 4, 9)
        wins = best_trades[best_trades > 0] * 100
        losses = best_trades[best_trades < 0] * 100

        box_plot = ax8.boxplot([wins, losses], labels=['Wins', 'Losses'], patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightgreen')
        box_plot['boxes'][1].set_facecolor('lightcoral')

        ax8.set_title(f' Win/Loss Analysis - {best_strategy}', fontweight='bold')
        ax8.set_ylabel('Return (%)')
        ax8.grid(True, alpha=0.3)

        # Add statistical annotations
        ax8.text(0.7, 0.95,
                 f'Wins: {len(wins)}\nMean: {wins.mean():.3f}%\nStd: {wins.std():.3f}%\nMax: {wins.max():.3f}%',
                 transform=ax8.transAxes, verticalalignment='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        ax8.text(0.7, 0.05,
                 f'Losses: {len(losses)}\nMean: {losses.mean():.3f}%\nStd: {losses.std():.3f}%\nMin: {losses.min():.3f}%',
                 transform=ax8.transAxes, verticalalignment='bottom', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

        # 9. Rolling Performance Analysis
        ax9 = plt.subplot(4, 4, 10)
        best_returns = best_equity.pct_change().dropna()
        window = min(252 * 75, len(best_returns) // 4)  # Adaptive window
        if window > 50:
            rolling_sharpe = (best_returns.rolling(window=window).mean() /
                              best_returns.rolling(window=window).std() * np.sqrt(250 * 375))
            rolling_sharpe.plot(ax=ax9, color='blue', alpha=0.7, linewidth=2)
            ax9.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Sharpe = 1.0')
            ax9.set_title(f' Rolling Sharpe Ratio - {best_strategy}', fontweight='bold')
            ax9.set_ylabel('Rolling Sharpe Ratio')
            ax9.legend()
        ax9.grid(True, alpha=0.3)

        # 10. Maximum Drawdown Comparison
        ax10 = plt.subplot(4, 4, 11)
        dd_data = self.dual_sma_results.head(8)
        colors_dd = ['red' if i == 0 else 'lightcoral' for i in range(len(dd_data))]
        bars = ax10.bar(range(len(dd_data)), dd_data['Max_Drawdown'] * 100,
                        color=colors_dd, alpha=0.8, edgecolor='black')

        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax10.text(bar.get_x() + bar.get_width() / 2., height - 0.5,
                      f'{height:.1f}%',
                      ha='center', va='top', fontweight='bold', color='white')

        ax10.set_title(' Max Drawdown Comparison (Top 8)', fontweight='bold')
        ax10.set_xlabel('Strategy Rank')
        ax10.set_ylabel('Max Drawdown (%)')
        ax10.set_xticks(range(len(dd_data)))
        ax10.set_xticklabels([row['Strategy'] for _, row in dd_data.iterrows()], rotation=45)
        ax10.grid(True, alpha=0.3)

        # 11. Volatility vs Trade Count Analysis
        ax11 = plt.subplot(4, 4, 12)
        scatter2 = ax11.scatter(self.dual_sma_results['Trade_Count'],
                                self.dual_sma_results['Volatility'] * 100,
                                c=self.dual_sma_results['Win_Rate'] * 100,
                                s=100, alpha=0.7, cmap='RdYlGn')
        plt.colorbar(scatter2, ax=ax11, label='Win Rate (%)')
        ax11.set_xlabel('Number of Trades')
        ax11.set_ylabel('Volatility (%)')
        ax11.set_title(' Trade Frequency vs Volatility', fontweight='bold')
        ax11.grid(True, alpha=0.3)

        # 12. Correlation Matrix of Key Metrics
        ax12 = plt.subplot(4, 4, (13, 14))
        key_metrics = ['Sharpe_Ratio', 'Calmar_Ratio', 'Win_Rate', 'Trade_Count', 'Volatility', 'Max_Drawdown']
        corr_matrix = self.dual_sma_results[key_metrics].corr()
        mask = np.triu(np.ones_like(corr_matrix))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', ax=ax12, cmap='RdBu_r',
                    center=0, cbar_kws={'label': 'Correlation'})
        ax12.set_title(' Metrics Correlation Matrix', fontweight='bold')

        # 13. Performance Summary Table
        ax13 = plt.subplot(4, 4, (15, 16))
        ax13.axis('off')

        # Create enhanced summary table
        summary_data = self.dual_sma_results.head(8)[['Strategy', 'Total_Return', 'Sharpe_Ratio',
                                                      'Calmar_Ratio', 'Max_Drawdown', 'Win_Rate', 'Trade_Count']]
        summary_data_display = summary_data.copy()
        summary_data_display['Total_Return'] = (summary_data_display['Total_Return'] * 100).round(1)
        summary_data_display['Sharpe_Ratio'] = summary_data_display['Sharpe_Ratio'].round(2)
        summary_data_display['Calmar_Ratio'] = summary_data_display['Calmar_Ratio'].round(2)
        summary_data_display['Max_Drawdown'] = (summary_data_display['Max_Drawdown'] * 100).round(1)
        summary_data_display['Win_Rate'] = (summary_data_display['Win_Rate'] * 100).round(1)

        table_text = summary_data_display.to_string(index=False)
        ax13.text(0.05, 0.95, ' TOP 8 DUAL SMA STRATEGIES SUMMARY:\n\n' + table_text,
                  transform=ax13.transAxes, fontsize=10, verticalalignment='top',
                  fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()
        plt.subplots_adjust(top=0.95, hspace=0.4, wspace=0.4)
        plt.show()

        print(" Comprehensive Dual SMA Dashboard created successfully!")

    def create_detailed_dual_sma_statistics(self):
        """Generate detailed statistical analysis and insights"""

        print("\n" + "=" * 80)
        print(" DETAILED DUAL SMA STATISTICAL ANALYSIS & INSIGHTS")
        print("=" * 80)

        # Overall performance summary
        print("\n TOP 10 DUAL SMA COMBINATIONS (Ranked by Sharpe Ratio):")
        print("-" * 100)
        display_cols = ['Strategy', 'Total_Return', 'Annualized_Return', 'Sharpe_Ratio',
                        'Calmar_Ratio', 'Max_Drawdown', 'Win_Rate', 'Trade_Count', 'Profit_Factor']
        display_df = self.dual_sma_results[display_cols].head(10).copy()

        # Format for better readability
        for col in ['Total_Return', 'Annualized_Return', 'Max_Drawdown', 'Win_Rate']:
            display_df[col] = (display_df[col] * 100).round(2)
        for col in ['Sharpe_Ratio', 'Calmar_Ratio', 'Profit_Factor']:
            display_df[col] = display_df[col].round(3)

        print(display_df.to_string(index=False))

        # Best strategy detailed analysis
        best_strategy_row = self.dual_sma_results.iloc[0]
        best_trades = self.dual_trade_data[self.best_dual_strategy]

        print(f"\n\n DETAILED ANALYSIS - BEST STRATEGY: {self.best_dual_strategy}")
        print("=" * 70)

        # Performance metrics
        print(f" PERFORMANCE METRICS:")
        print(f"├─ Total Return: {best_strategy_row['Total_Return'] * 100:.2f}%")
        print(f"├─ Annualized Return: {best_strategy_row['Annualized_Return'] * 100:.2f}%")
        print(f"├─ Volatility: {best_strategy_row['Volatility'] * 100:.2f}%")
        print(f"├─ Sharpe Ratio: {best_strategy_row['Sharpe_Ratio']:.3f}")
        print(f"├─ Calmar Ratio: {best_strategy_row['Calmar_Ratio']:.3f}")
        print(f"├─ Sortino Ratio: {best_strategy_row['Sortino_Ratio']:.3f}")
        print(f"└─ Maximum Drawdown: {best_strategy_row['Max_Drawdown'] * 100:.2f}%")

        # Trade statistics
        print(f"\n TRADE STATISTICS:")
        print(f"├─ Total Trades: {best_strategy_row['Trade_Count']}")
        print(f"├─ Win Rate: {best_strategy_row['Win_Rate'] * 100:.2f}%")
        print(f"├─ Average Winning Trade: {best_strategy_row['Avg_Win'] * 100:.3f}%")
        print(f"├─ Average Losing Trade: {best_strategy_row['Avg_Loss'] * 100:.3f}%")
        print(f"├─ Profit Factor: {best_strategy_row['Profit_Factor']:.3f}")
        print(f"├─ Win/Loss Ratio: {-best_strategy_row['Avg_Win'] / best_strategy_row['Avg_Loss']:.3f}")

        # Distribution statistics
        print(f"\n RETURN DISTRIBUTION STATISTICS:")
        print(f"├─ Mean Trade Return: {best_trades.mean() * 100:.4f}%")
        print(f"├─ Median Trade Return: {best_trades.median() * 100:.4f}%")
        print(f"├─ Standard Deviation: {best_trades.std() * 100:.4f}%")
        print(f"├─ Skewness: {best_strategy_row['Trade_Skew']:.3f}")
        print(f"├─ Kurtosis: {best_strategy_row['Trade_Kurt']:.3f}")
        print(f"├─ Shapiro-Wilk p-value: {best_strategy_row['Shapiro_P']:.4f}")
        print(f"└─ Jarque-Bera p-value: {best_strategy_row['JB_P']:.4f}")

        # Percentile analysis
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        print(f"\n TRADE RETURN PERCENTILES:")
        for p in percentiles:
            pct_val = np.percentile(best_trades, p)
            print(f"├─ {p:2d}th percentile: {pct_val * 100:.3f}%")

        # Tail analysis
        tail_5_pct = best_trades.quantile(0.95)
        tail_trades = best_trades[best_trades >= tail_5_pct]
        tail_contribution = tail_trades.sum() / best_trades.sum() * 100 if best_trades.sum() != 0 else 0

        print(f"\n TAIL RISK ANALYSIS:")
        print(f"├─ Top 5% trade threshold: {tail_5_pct * 100:.3f}%")
        print(f"├─ Top 5% trades contribute: {tail_contribution:.1f}% of total profit")
        print(f"├─ Number of tail trades: {len(tail_trades)}")
        print(f"└─ Average tail trade: {tail_trades.mean() * 100:.3f}%")

        # Win/Loss streak analysis
        wins_losses = (best_trades > 0).astype(int)
        streaks = []
        current_streak = 1
        current_type = wins_losses.iloc[0] if len(wins_losses) > 0 else 0

        for i in range(1, len(wins_losses)):
            if wins_losses.iloc[i] == current_type:
                current_streak += 1
            else:
                streaks.append((current_type, current_streak))
                current_type = wins_losses.iloc[i]
                current_streak = 1
        streaks.append((current_type, current_streak))

        win_streaks = [length for type_val, length in streaks if type_val == 1]
        loss_streaks = [length for type_val, length in streaks if type_val == 0]

        print(f"\n STREAK ANALYSIS:")
        print(f"├─ Maximum winning streak: {max(win_streaks) if win_streaks else 0} trades")
        print(
            f"├─ Average winning streak: {np.mean(win_streaks):.1f} trades" if win_streaks else "├─ Average winning streak: 0 trades")
        print(f"├─ Maximum losing streak: {max(loss_streaks) if loss_streaks else 0} trades")
        print(
            f"└─ Average losing streak: {np.mean(loss_streaks):.1f} trades" if loss_streaks else "└─ Average losing streak: 0 trades")

        # Market regime analysis
        print(f"\n MARKET REGIME INSIGHTS:")
        best_equity = self.dual_equity_curves[self.best_dual_strategy]
        monthly_returns = best_equity.resample('M').last().pct_change().dropna()
        if len(monthly_returns) > 0:
            positive_months = (monthly_returns > 0).sum()
            total_months = len(monthly_returns)
            print(f"├─ Positive months: {positive_months}/{total_months} ({positive_months / total_months * 100:.1f}%)")
            print(f"├─ Best month: {monthly_returns.max() * 100:.2f}%")
            print(f"├─ Worst month: {monthly_returns.min() * 100:.2f}%")
            print(f"└─ Monthly volatility: {monthly_returns.std() * 100:.2f}%")
        return best_strategy_row, best_trades

    def advanced_dual_sma_insights(self):
        """Generate additional advanced insights"""

        print(f"\n ADVANCED DUAL SMA INSIGHTS")
        print("=" * 50)

        # Fast vs Slow SMA analysis
        fast_performance = self.dual_sma_results.groupby('Fast_SMA').agg({
            'Sharpe_Ratio': 'mean',
            'Total_Return': 'mean',
            'Win_Rate': 'mean'
        }).round(3)

        slow_performance = self.dual_sma_results.groupby('Slow_SMA').agg({
            'Sharpe_Ratio': 'mean',
            'Total_Return': 'mean',
            'Win_Rate': 'mean'
        }).round(3)

        print(f"\n FAST SMA PERIOD ANALYSIS:")
        print(fast_performance)

        print(f"\n SLOW SMA PERIOD ANALYSIS:")
        print(slow_performance)

        # Best fast and slow periods
        best_fast = fast_performance['Sharpe_Ratio'].idxmax()
        best_slow = slow_performance['Sharpe_Ratio'].idxmax()

        print(f"\n OPTIMAL PERIODS:")
        print(f"├─ Best Fast SMA Period: {best_fast}")
        print(f"└─ Best Slow SMA Period: {best_slow}")

        return fast_performance, slow_performance

def main():
    backtester=NiftyBacktester()
    df=backtester.load_and_prepare_data()
    backtester.exploratory_data_analysis()
    df=backtester.feature_engineering()
    backtester.run_backtest()
    backtester.create_performance_visualizations()
    backtester.statistical_analysis_best_sma()
    backtester.comprehensive_dual_sma_crossover_analysis()
    backtester.create_dual_sma_comprehensive_dashboard()
    backtester.create_detailed_dual_sma_statistics()
    backtester.advanced_dual_sma_insights()


if __name__ == "__main__":
    main()
    print("Backtesting and analysis completed successfully!")



