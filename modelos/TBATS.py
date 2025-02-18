import pandas as pd
import numpy as np
from tbats import TBATS
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

class TBATSModel:
    def __init__(self, seasonal_periods: List[int] = [5, 22],  # 5 dias (semana), 22 dias (mês)
                 use_box_cox: bool = True,
                 use_trend: bool = True,
                 use_damped_trend: bool = False):
        """
        Initialize TBATS model with specified parameters
        
        Args:
            seasonal_periods: List of seasonal periods to consider
            use_box_cox: Whether to use Box-Cox transformation
            use_trend: Whether to include trend component
            use_damped_trend: Whether to use damped trend
        """
        self.seasonal_periods = seasonal_periods
        self.use_box_cox = use_box_cox
        self.use_trend = use_trend
        self.use_damped_trend = use_damped_trend
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
        Prepare data for TBATS modeling
        
        Args:
            data: Original time series data
            
        Returns:
            Transformed time series data
        """
        # Remove missing values
        data = data.dropna()
        
        # Check stationarity and apply transformations if needed
        is_stationary, transformed_data = self.check_stationarity(data)
        
        return transformed_data
    
    def train(self, data: pd.Series):
        """
        Train the TBATS model
        
        Args:
            data: Time series data for training
        """
        # Store original data
        self.original_data = data
        
        # Prepare data
        prepared_data = self.prepare_data(data)
        
        # Create and fit TBATS model
        self.model = TBATS(
            seasonal_periods=self.seasonal_periods,
            use_box_cox=self.use_box_cox,
            use_trend=self.use_trend,
            use_damped_trend=self.use_damped_trend
        )
        
        print("\nFitting TBATS model...")
        print(f"Seasonal periods: {self.seasonal_periods}")
        print(f"Using Box-Cox transformation: {self.use_box_cox}")
        print(f"Using trend: {self.use_trend}")
        print(f"Using damped trend: {self.use_damped_trend}")
        
        self.fitted_model = self.model.fit(prepared_data)
        
        # Print model summary
        print("\nModel components:")
        print(f"Box-Cox transformation parameter: {self.fitted_model.params.box_cox_lambda}")
        print(f"Alpha (Level) parameter: {self.fitted_model.params.alpha}")
        print(f"Beta (Trend) parameter: {self.fitted_model.params.beta}")
        
        # Print seasonal parameters
        params = self.fitted_model.params
        for i, period in enumerate(self.seasonal_periods, 1):
            try:
                gamma_value = getattr(params, f'_gamma_{i}', None)
                if gamma_value is not None:
                    print(f"Gamma_{i} (Seasonal) parameter for period {period}: {gamma_value}")
            except AttributeError:
                print(f"No gamma parameter found for period {period}")
        
        if hasattr(self.fitted_model.params, 'phi'):
            print(f"Phi (Damping) parameter: {self.fitted_model.params.phi}")
        
        # Print additional model information
        print("\nModel Information:")
        print(f"Number of seasonal patterns: {len(self.seasonal_periods)}")
        print(f"Seasonal periods used: {self.seasonal_periods}")
        
        # Print model fit statistics if available
        try:
            aic = getattr(self.fitted_model, 'aic', None)
            bic = getattr(self.fitted_model, 'bic', None)
            
            if aic is not None and not callable(aic):
                print(f"AIC: {aic}")
            if bic is not None and not callable(bic):
                print(f"BIC: {bic}")
        except Exception as e:
            print("Note: AIC/BIC statistics not available")
    
    def predict(self, n_steps: int, start_idx: Optional[int] = None) -> pd.Series:
        """
        Make predictions for n steps ahead
        
        Args:
            n_steps: Number of steps to forecast
            start_idx: Optional start index for in-sample predictions
            
        Returns:
            Series with predictions
        """
        if self.fitted_model is None:
            raise ValueError("Model must be trained before making predictions")
        
        if start_idx is not None:
            # In-sample prediction
            # TBATS doesn't support direct in-sample prediction, so we refit the model
            train_data = self.original_data[:start_idx]
            temp_model = TBATS(
                seasonal_periods=self.seasonal_periods,
                use_box_cox=self.use_box_cox,
                use_trend=self.use_trend,
                use_damped_trend=self.use_damped_trend
            ).fit(train_data)
            
            forecast = temp_model.forecast(steps=n_steps)
        else:
            # Out-of-sample forecast
            forecast = self.fitted_model.forecast(steps=n_steps)
        
        # Create index for forecast
        if start_idx is not None:
            last_date = self.original_data.index[start_idx-1]
        else:
            last_date = self.original_data.index[-1]
            
        forecast_index = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=n_steps,
            freq='B'  # Business days
        )
        
        return pd.Series(forecast, index=forecast_index)
    
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
    
    def plot_predictions(self, train_data: pd.Series, predictions: pd.Series,
                        test_data: Optional[pd.Series] = None, title: str = None):
        """
        Plot the training data, predictions and test data if available
        
        Args:
            train_data: Training data
            predictions: Model predictions
            test_data: Test data (optional)
            title: Custom title for the plot
        """
        fig = plt.figure(figsize=(15, 8))
        
        # Plot training data
        plt.plot(train_data.index[-100:], train_data.values[-100:], 
                label='Training Data', color='blue', alpha=0.7)
        
        # Plot predictions
        plt.plot(predictions.index, predictions.values, 
                label='Predictions', color='red', linestyle='--')
        
        if test_data is not None:
            # Ensure we're only plotting test data for the prediction period
            test_period = test_data[predictions.index[0]:predictions.index[-1]]
            plt.plot(test_period.index, test_period.values, 
                    label='Actual Values', color='green', alpha=0.7)
            
            # Add error bands only if we have matching predictions and test data
            if len(predictions) == len(test_period):
                error = np.abs(predictions - test_period)
                plt.fill_between(predictions.index,
                               predictions - error,
                               predictions + error,
                               color='red', alpha=0.1,
                               label='Error Margin')
        
        plt.title(title or 'TBATS Time Series Forecast')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Show and close the figure properly
        plt.show()
        plt.close(fig)

    def backtest(self, data: pd.Series, start_idx: int, n_steps: int):
        """
        Perform backtesting starting from a specific index
        
        Args:
            data: Complete time series data
            start_idx: Start index for backtesting
            n_steps: Number of steps to forecast
        """
        # Get predictions
        predictions = self.predict(n_steps, start_idx=start_idx)
        
        # Get actual values for the same period
        actual_values = data[predictions.index]
        
        # Calculate metrics
        metrics = self.evaluate(actual_values, predictions)
        
        # Get the date for the title
        start_date = data.index[start_idx].strftime('%Y-%m-%d')
        
        # Plot results
        self.plot_predictions(
            train_data=data[:start_idx],
            predictions=predictions,
            test_data=actual_values,
            title=f'TBATS Backtest - {n_steps} Steps from {start_date}'
        )
        
        return predictions, actual_values, metrics

def main():
    # Get the absolute path to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    file_path = os.path.join(project_root, 'dados', 'raw', 'PETR4.SA.csv')
    
    print(f"Loading data from: {file_path}")
    
    # Initialize model
    model = TBATSModel(
        seasonal_periods=[5, 22],  # Semana de trading e mês de trading
        use_box_cox=True,         # Transformação para estabilizar variância
        use_trend=True,           # Incluir componente de tendência
        use_damped_trend=False    # Sem amortecimento da tendência
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
        
        model.plot_predictions(
            train_data, 
            predictions, 
            test_data[:horizon] if horizon <= len(test_data) else None,
            title=f'Future Forecast - {horizon} Days Ahead'
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
        predictions, actuals, metrics = model.backtest(data, start_idx, n_steps=30)
        
        print("Backtest metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main() 