import os
import numpy as np
import pandas as pd
import joblib
import json
from tensorflow.keras.models import load_model
from .LSTM import LSTMModel
from tensorflow.keras.losses import MeanSquaredError
from typing import Union, Tuple

# Get the absolute path to the module's directory
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINED_DIR = os.path.join(MODULE_DIR, 'trained')

def ensure_trained_dir():
    """Ensure the trained models directory exists"""
    os.makedirs(TRAINED_DIR, exist_ok=True)

def get_model_path(stock_name: str) -> str:
    """Get the path to a model's directory"""
    stock_name_clean = stock_name.split('.')[0]
    return os.path.join(TRAINED_DIR, stock_name_clean)

def predict_with_saved_model(model_dir: str, input_sequence=None, forecast_days: int = 1) -> np.array:
    """
    Load a saved LSTM model and generate predictions.
    
    Args:
        model_dir (str): Directory containing saved model artifacts
        input_sequence (np.array, optional): Input sequence for prediction
        forecast_days (int): Number of days to forecast
    
    Returns:
        np.array: Predicted values
    """
    # Load model configuration
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    try:
        # Try loading the model normally
        model = load_model(os.path.join(model_dir, 'lstm_model.h5'))
    except ValueError as e:
        if "Could not locate function 'mse'" in str(e):
            # If the error is about 'mse', try loading with custom_objects
            model = load_model(
                os.path.join(model_dir, 'lstm_model.h5'),
                custom_objects={'loss': MeanSquaredError()}
            )
        else:
            raise e
    
    scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
    
    # Get time steps from configuration
    time_steps = config.get('time_steps', 10)
    
    # If no input sequence is provided, raise an error
    if input_sequence is None:
        raise ValueError("Input sequence must be provided for prediction.")
    
    # Ensure input sequence has correct shape
    if input_sequence.ndim == 1:
        input_sequence = input_sequence.reshape(1, -1, 1)
    elif input_sequence.ndim == 2:
        input_sequence = input_sequence.reshape(1, input_sequence.shape[0], 1)
    
    # Validate input sequence shape
    if input_sequence.shape[1] != time_steps:
        raise ValueError(f"Input sequence must have {time_steps} time steps")
    
    # Generate predictions
    predictions = []
    current_sequence = input_sequence.copy()
    
    for _ in range(forecast_days):
        # Predict next value
        next_pred = model.predict(current_sequence, verbose=0)[0]
        
        # Inverse transform the prediction
        next_pred_original_scale = scaler.inverse_transform(next_pred.reshape(-1, 1))[0][0]
        predictions.append(next_pred_original_scale)
        
        # Update sequence for next prediction
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, 0] = next_pred
    
    return np.array(predictions)

def get_model_prediction(stock_name: str, df: pd.DataFrame, return_plot: bool = False) -> Union[np.array, Tuple[np.array, str]]:
    """
    Get predictions for a stock using a saved model or train a new one.
    
    Args:
        stock_name: Name of the stock
        df: DataFrame with stock data
        return_plot: Whether to return a base64 encoded plot
        
    Returns:
        Union[np.array, Tuple[np.array, str]]: Predictions and optionally the plot
    """
    ensure_trained_dir()
    model_dir = get_model_path(stock_name)
    
    try:
        # Try to load existing model
        scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
    except (FileNotFoundError, OSError):
        # Train new model if loading fails
        print(f"Training new model for {stock_name}")
        model = LSTMModel()
        model.train(df['Close'])
        model.save_model(model_dir)
        scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
    
    # Prepare the last sequence
    last_sequence = df['Close'].values[-10:]  # Using default 10 time steps
    scaled_sequence = scaler.transform(last_sequence.reshape(-1, 1)).reshape(1, -1, 1)
    
    # Make predictions
    predictions = predict_with_saved_model(
        model_dir=model_dir,
        input_sequence=scaled_sequence,
        forecast_days=7
    )
    
    if return_plot:
        plot = generate_prediction_plot(stock_name, df, predictions)
        return predictions, plot
    
    return predictions

def list_trained_models() -> list:
    """List all trained models available"""
    ensure_trained_dir()
    return [d for d in os.listdir(TRAINED_DIR) 
            if os.path.isdir(os.path.join(TRAINED_DIR, d))]

def generate_prediction_plot(stock_name: str, df: pd.DataFrame, predictions: np.array) -> str:
    """
    Generate a plot showing historical data and predictions
    
    Args:
        stock_name: Name of the stock
        df: DataFrame with historical data
        predictions: Array of predicted values
        
    Returns:
        str: Base64 encoded plot image
    """
    import plotly.graph_objs as go
    import plotly.io as pio
    import base64
    from datetime import datetime
    
    # Create future dates for predictions
    last_date = df.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=len(predictions),
        freq='B'  # Business days frequency
    )
    
    # Get last 100 days of historical data
    historical_cutoff = last_date - pd.Timedelta(days=100)
    historical_data = df[df.index >= historical_cutoff]['Close']
    
    # Create figure
    fig = go.Figure()
    
    # Historical data trace
    fig.add_trace(go.Scatter(
        x=historical_data.index,
        y=historical_data.values,
        mode='lines',
        name='Historical Data',
        line=dict(color='blue')
    ))
    
    # Predictions trace
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=predictions,
        mode='lines',
        name='Predictions',
        line=dict(color='red', dash='dash', width=2)
    ))
    
    # Update layout
    fig.update_layout(
        title=f'{stock_name} Price Forecast',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x unified',
        legend_title_text='Data Types',
        showlegend=True
    )
    
    # Configure x-axis to show days
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGray',
        tickformat='%d/%m/%y',  # Show as DD/MM/YY
        dtick='D1',  # Show daily ticks
        tickangle=45
    )
    
    # Configure y-axis
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGray'
    )
    
    # Convert to static image and encode in base64
    img_bytes = pio.to_image(fig, format="png", scale=2)
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
    return img_base64