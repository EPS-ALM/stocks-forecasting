import os
import numpy as np
import pandas as pd
import joblib
import json
from tensorflow.keras.models import load_model
from LSTM import *

def predict_with_saved_model(model_dir='model_artifacts', input_sequence=None, forecast_days=1):
    """
    Load a saved LSTM model and generate predictions.
    
    Args:
        model_dir (str): Directory containing saved model artifacts
        input_sequence (np.array, optional): Input sequence for prediction. 
                                             If None, uses the last sequence from the original data.
        forecast_days (int): Number of days to forecast
    
    Returns:
        np.array: Predicted values
    """
    # Load model configuration
    config_path = os.path.join(model_dir, 'model_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load model and scaler
    model = load_model(os.path.join(model_dir, 'lstm_model.h5'))
    scaler = joblib.load(os.path.join(model_dir, 'data_scaler.joblib'))
    
    # Get time steps from configuration
    time_steps = config.get('time_steps', 10)
    
    # If no input sequence is provided, raise an error
    if input_sequence is None:
        raise ValueError("Input sequence must be provided for prediction. "
                         "Ensure it has shape (1, time_steps, 1) and is scaled.")
    
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
        next_pred = model.predict(current_sequence)[0]
        
        # Inverse transform the prediction
        next_pred_original_scale = scaler.inverse_transform(next_pred.reshape(-1, 1))[0][0]
        predictions.append(next_pred_original_scale)
        
        # Update sequence for next prediction
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, 0] = next_pred
    
    return np.array(predictions)

# Example usage function
def get_model_prediction(nome_acao:str, df:pd.DataFrame):
    
    """
    Demonstrate how to use the prediction function with a saved model.
    """
    # Load original data (for getting the last sequence)
    #df = pd.read_csv('stock_data.csv', parse_dates=['Date'], index_col='Date')
    
    nome_acao_tratado = nome_acao.split('.')[0]
    
    try:
        # Load scaler from saved model artifacts
        scaler = joblib.load(f'trained/{nome_acao_tratado}/data_scaler.joblib')
    except:
        run_model_training_workflow(df, nome_acao)
        # Load scaler from saved model artifacts
        scaler = joblib.load(f'trained/{nome_acao_tratado}/data_scaler.joblib')
    
    # Prepare the last sequence from original data
    last_sequence = df['Close'].values[-10:]  # Assuming 10 time steps
    
    # Scale the input sequence
    scaled_sequence = scaler.transform(last_sequence.reshape(-1, 1)).reshape(1, -1, 1)
    
    # Make predictions
    predictions = predict_with_saved_model(
        model_dir=f'trained/{nome_acao_tratado}', 
        input_sequence=scaled_sequence, 
        forecast_days=7
    )
    
    return predictions