import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import warnings
import os
from datetime import datetime
import matplotlib
import logging
import io
import base64

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

matplotlib.use('Agg')  # Use non-GUI backend

class SARIMAModel:
    def __init__(self, order: Tuple[int, int, int] = (2, 1, 2),
                 seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 5)):
        """
        Initialize SARIMA model with specified parameters
        
        Args:
            order: ARIMA order (p, d, q)
            seasonal_order: Seasonal order (P, D, Q, s)
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
        
    def check_stationarity(self, data: pd.Series) -> Tuple[bool, pd.Series]:
        """
        Check if the time series is stationary using ADF test and make it stationary if needed
        
        Args:
            data: Time series data
            
        Returns:
            Tuple of (is_stationary, transformed_data)
        """
        # Primeira verificação
        result = adfuller(data.values)
        is_stationary = result[1] < 0.05
        
        if is_stationary:
            return True, data
            
        # Tenta diferenciação
        diff_data = data.diff().dropna()
        result = adfuller(diff_data.values)
        is_stationary = result[1] < 0.05
        
        if is_stationary:
            print("Series became stationary after first difference")
            return True, diff_data
            
        # Tenta diferenciação de segunda ordem
        diff2_data = diff_data.diff().dropna()
        result = adfuller(diff2_data.values)
        is_stationary = result[1] < 0.05
        
        if is_stationary:
            print("Series became stationary after second difference")
            return True, diff2_data
            
        print("Warning: Series remains non-stationary even after transformations")
        return False, data
    
    def load_data(self, file_path: str, target_column: str = 'Adj Close') -> pd.DataFrame:
        """
        Load and prepare time series data
        
        Args:
            file_path: Path to the CSV file
            target_column: Column to be used for predictions
            
        Returns:
            Processed DataFrame with datetime index
        """
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df[target_column]
    
    def prepare_data(self, data: pd.Series) -> pd.Series:
        """
        Prepare data for SARIMA modeling by applying necessary transformations
        
        Args:
            data: Original time series data
            
        Returns:
            Transformed time series data
        """
        # Apply returns transformation
        data_returns = np.log(data / data.shift(1)).dropna()
        
        # Check stationarity and apply additional transformations if needed
        is_stationary, transformed_data = self.check_stationarity(data_returns)
        
        return transformed_data
    
    def inverse_transform(self, data: pd.Series, original_data: pd.Series) -> pd.Series:
        """
        Inverse transform the predictions back to original scale
        
        Args:
            data: Transformed predictions
            original_data: Original data series for reference
            
        Returns:
            Predictions in original scale
        """
        # Get the last value from the original data
        last_value = original_data.iloc[-1]
        
        # Convert from returns to prices
        predicted_prices = last_value * np.exp(data.cumsum())
        
        return predicted_prices
    
    def train(self, data: pd.Series):
        """
        Train the SARIMA model
        
        Args:
            data: Time series data for training
        """
        # Store original data for inverse transform
        self.original_data = data
        
        # Prepare data
        prepared_data = self.prepare_data(data)
        
        # Create and fit model with more robust parameters
        self.model = SARIMAX(
            prepared_data,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        self.fitted_model = self.model.fit(
            disp=False,
            maxiter=1000,
            method='lbfgs'
        )
        
    def predict(self, stock_data: pd.Series, n_steps: int, start_idx: Optional[int] = None) -> pd.Series:
        """
        Make predictions for n steps ahead
        """
        if self.fitted_model is None:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            logger.debug("Starting prediction process...")
            logger.debug(f"Input stock_data type: {type(stock_data)}")
            logger.debug(f"Input n_steps: {n_steps}")
            logger.debug(f"Input start_idx: {start_idx}")
            
            if start_idx is not None:
                # In-sample prediction
                forecast = self.fitted_model.get_prediction(start=start_idx, dynamic=True)
                predictions = forecast.predicted_mean
            else:
                # Out-of-sample forecast
                forecast = self.fitted_model.forecast(steps=n_steps)
                
                # Create future dates index
                last_date = stock_data.index[-1]
                future_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=n_steps,
                    freq='B'  # Business days frequency
                )
                
                # Create series with future dates index
                predictions = pd.Series(forecast, index=future_dates)
            
            logger.debug(f"Raw predictions type: {type(predictions)}")
            logger.debug(f"Raw predictions:\n{predictions}")
            
            # Transform predictions back to original scale
            predictions_transformed = self.inverse_transform(predictions, self.original_data)
            
            logger.debug(f"Transformed predictions type: {type(predictions_transformed)}")
            logger.debug(f"Transformed predictions:\n{predictions_transformed}")
            
            # Ensure we have a pandas Series with float values
            if not isinstance(predictions_transformed, pd.Series):
                predictions_transformed = pd.Series(predictions_transformed, index=predictions.index)
            predictions_transformed = predictions_transformed.astype(float)
            
            logger.debug(f"Final predictions shape: {predictions_transformed.shape}")
            logger.debug(f"Final predictions dtype: {predictions_transformed.dtype}")
            logger.debug(f"Final predictions head:\n{predictions_transformed.head()}")
            
            return predictions_transformed
            
        except Exception as e:
            logger.error(f"Error in predict method: {str(e)}")
            raise

    def generate_plot(self, stock_data: pd.Series, predictions: pd.Series, name: Optional[str] = None) -> str:
        """
        Generate plot for the predictions
        """
        try:
            logger.debug("Starting plot generation...")
            logger.debug(f"Stock data type: {type(stock_data)}")
            logger.debug(f"Stock data index type: {type(stock_data.index)}")
            logger.debug(f"Stock data values type: {type(stock_data.values)}")
            logger.debug(f"Stock data shape: {stock_data.values.shape}")
            logger.debug(f"Predictions type: {type(predictions)}")
            logger.debug(f"Predictions index type: {type(predictions.index)}")
            logger.debug(f"Predictions values type: {type(predictions.values)}")
            logger.debug(f"Predictions shape: {predictions.values.shape}")
            
            # Ensure we have pandas Series with 1D data
            if not isinstance(stock_data, pd.Series):
                stock_data = pd.Series(stock_data.ravel() if hasattr(stock_data, 'ravel') else stock_data)
            if not isinstance(predictions, pd.Series):
                predictions = pd.Series(predictions.ravel() if hasattr(predictions, 'ravel') else predictions)
            
            # Clear any existing plots
            plt.clf()
            plt.close('all')
            
            # Create new figure with adjusted size for better readability
            plt.figure(figsize=(15, 8))
            
            # Get the last date from historical data
            last_date = stock_data.index[-1]
            logger.debug(f"Last date: {last_date}")
            
            # Calculate cutoff dates
            historical_cutoff = last_date - pd.Timedelta(days=15)
            forecast_cutoff = last_date + pd.Timedelta(days=15)
            logger.debug(f"Historical cutoff: {historical_cutoff}")
            logger.debug(f"Forecast cutoff: {forecast_cutoff}")
            
            # Filter data for the last 15 days
            historical_data = stock_data[stock_data.index >= historical_cutoff]
            
            # Ensure predictions are also within the 15-day window
            # Convert predictions index to pandas datetime if it's not already
            if not isinstance(predictions.index, pd.DatetimeIndex):
                predictions.index = pd.to_datetime(predictions.index)
            predictions = predictions[predictions.index <= forecast_cutoff]
            
            historical_values = historical_data.values.ravel().astype(np.float64)
            prediction_values = predictions.values.ravel().astype(np.float64)
            
            logger.debug(f"Historical data shape: {historical_data.shape}")
            logger.debug(f"Historical values shape: {historical_values.shape}")
            logger.debug(f"Prediction values shape: {prediction_values.shape}")
            
            # Convert datetime index to matplotlib dates
            historical_dates = historical_data.index.astype('datetime64[ns]')
            prediction_dates = predictions.index.astype('datetime64[ns]')
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # Plot with improved formatting
            ax.plot(
                historical_dates,
                historical_values,
                label='Historical Data',
                color='blue',
                linewidth=2
            )
            
            ax.plot(
                prediction_dates,
                prediction_values,
                label='Forecast',
                color='red',
                linestyle='--',
                linewidth=2
            )
            
            # Set title with ticker name if provided
            title = str(f'Stock Price Forecast - {name}' if name else 'Stock Price Forecast')
            ax.set_title(title, fontsize=14, pad=20)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Price', fontsize=12)
            
            # Improve x-axis formatting with AutoDateLocator and fixed number of ticks
            locator = plt.matplotlib.dates.AutoDateLocator(maxticks=15)
            formatter = plt.matplotlib.dates.DateFormatter('%Y-%m-%d')
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            
            # Rotate and align the tick labels so they look better
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Add grid for better readability
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add legend with better positioning
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout()
            
            logger.debug("Saving plot to buffer...")
            
            # Save plot to bytes buffer with higher DPI for better quality
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            
            # Convert to base64
            image_bytes = buffer.getvalue()
            buffer.close()
            plt.close('all')  # Close all figures
            
            # Convert bytes to base64 string
            image_base64 = base64.b64encode(image_bytes)
            result = image_base64.decode('utf-8')
            
            logger.debug(f"Generated base64 string length: {len(result)}")
            return result
            
        except Exception as e:
            logger.error(f"Error in generate_plot: {str(e)}")
            plt.close('all')  # Ensure all figures are closed even if error occurs
            raise Exception(f"Error generating plot: {str(e)}")

    def evaluate(self, y_true: pd.Series, y_pred: pd.Series) -> dict:
        """
        Evaluate model performance
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary with evaluation metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
    
    def backtest(self, data: pd.Series, start_idx: int, n_steps: int):
        """
        Perform backtesting starting from a specific index
        
        Args:
            data: Complete time series data
            start_idx: Start index for backtesting
            n_steps: Number of steps to forecast
        """
        # Get predictions
        predictions = self.predict(data, n_steps, start_idx=start_idx)
        
        # Get actual values for the same period
        actual_values = data[predictions.index]
        
        # Calculate metrics
        metrics = self.evaluate(actual_values, predictions)
        
        # Generate plot
        base64_plot = self.generate_plot(data, predictions)
        
        return predictions, actual_values, metrics, base64_plot

    def calculate_metrics(self, data: pd.Series) -> dict:
        """
        Calculate basic metrics for the model using in-sample predictions
        
        Args:
            data: Original time series data
            
        Returns:
            Dictionary containing basic metrics
        """
        try:
            # Get in-sample predictions
            in_sample_preds = self.predict(data, n_steps=len(data), start_idx=0)
            
            # Calculate metrics
            mse = mean_squared_error(data, in_sample_preds)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(data, in_sample_preds)
            
            # Calculate MAPE
            mape = np.mean(np.abs((data - in_sample_preds) / data)) * 100
            
            return {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'mape': float(mape)
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {
                'mse': None,
                'rmse': None,
                'mae': None,
                'mape': None
            }

def main():
    # Get the absolute path to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    file_path = os.path.join(project_root, 'dados', 'raw', 'PETR4.SA.csv')
    
    print(f"Loading data from: {file_path}")
    
    # Initialize model with better parameters for financial data
    model = SARIMAModel(
        order=(2, 1, 2),           # (p, d, q) - Aumentei a ordem para capturar mais padrões
        seasonal_order=(1, 1, 1, 5) # (P, D, Q, s) - Período sazonal menor para dados financeiros
    )
    
    # Load data
    data = model.load_data(file_path)
    
    # Print data range
    print(f"\nData range: from {data.index.min()} to {data.index.max()}")
    
    # Split data into train and test
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # Train model
    print("\nTraining model...")
    model.train(train_data)
    
    # 1. Future predictions
    prediction_horizons = [1, 5, 7, 15, 30]
    for horizon in prediction_horizons:
        print(f"\nPredicting {horizon} days ahead (future):")
        predictions = model.predict(horizon)
        
        if horizon <= len(test_data):
            metrics = model.evaluate(test_data[:horizon], predictions)
            print("Evaluation metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
        
        model.generate_plot(
            data, 
            predictions,
            name=f'Future Forecast - {horizon} Days Ahead'
        )
    
    # 2. Backtesting (in-sample predictions)
    # Get indices for backtesting
    total_days = len(data)
    backtest_indices = [
        int(total_days * 0.5),  # Middle of the dataset
        int(total_days * 0.6),  # 60% through the dataset
        int(total_days * 0.7)   # 70% through the dataset
    ]
    
    # Print the actual dates we'll use for backtesting
    print("\nBacktesting dates:")
    for idx in backtest_indices:
        print(f"Index {idx}: {data.index[idx].strftime('%Y-%m-%d')}")
    
    for start_idx in backtest_indices:
        date = data.index[start_idx].strftime('%Y-%m-%d')
        print(f"\nBacktesting from {date} (index {start_idx}):")
        predictions, actuals, metrics, plot = model.backtest(data, start_idx, n_steps=30)
        
        print("Backtest metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main() 