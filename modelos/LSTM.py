import os
import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam


def get_avaible_models():
    return os.listdir('./modelos/trained')

def list_repository_files(repository_path):
    """Lists all files within a given repository.

    Args:
        repository_path (str): The path to the repository or directory.

    Returns:
        list: A list containing paths to all files in the repository.
    """
    files = []
    for root, dirs, file_names in os.walk(repository_path):
        for file_name in file_names:
            # Adds the full file path
            files.append(os.path.join(root, file_name))

    return files

def prepare_data(data, time_steps=1):
    """
    Prepare data for LSTM by creating sliding window sequences.
    
    Args:
        data (np.array): Input data series
        time_steps (int): Number of previous time steps to use for prediction
    
    Returns:
        X (np.array): Input sequences
        y (np.array): Target values
    """
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

def create_lstm_model(input_shape, units=50):
    """
    Create LSTM model for time series forecasting.
    
    Args:
        input_shape (tuple): Shape of input data
        units (int): Number of LSTM units
    
    Returns:
        model (keras.Model): Compiled LSTM model
    """
    model = Sequential([
        LSTM(units, activation='relu', input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def save_model_artifacts(model, scaler, time_steps, save_dir='model_artifacts'):
    """
    Save model, scaler, and configuration for later use.
    
    Args:
        model (keras.Model): Trained LSTM model
        scaler (MinMaxScaler): Fitted data scaler
        time_steps (int): Number of time steps used in training
        save_dir (str): Directory to save model artifacts
    
    Returns:
        str: Path to the saved model directory
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save Keras model
    model_path = os.path.join(save_dir, 'lstm_model.h5')
    model.save(model_path)
    
    # Save scaler
    scaler_path = os.path.join(save_dir, 'data_scaler.joblib')
    joblib.dump(scaler, scaler_path)
    
    # Save configuration
    config_path = os.path.join(save_dir, 'model_config.json')
    config = {
        'time_steps': time_steps,
        'input_shape': list(model.input_shape),
        'model_path': model_path,
        'scaler_path': scaler_path
    }
    import json
    with open(config_path, 'w') as f:
        json.dump(config, f)
    
    print(f"Model artifacts saved to {save_dir}")
    return save_dir

def load_model_artifacts(model_dir='model_artifacts'):
    """
    Load saved model artifacts for forecasting.
    
    Args:
        model_dir (str): Directory containing saved model artifacts
    
    Returns:
        dict: Dictionary containing loaded model, scaler, and configuration
    """
    # Load configuration
    with open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
        config = json.load(f)
    
    # Load model
    model = load_model(config['model_path'])
    
    # Load scaler
    scaler = joblib.load(config['scaler_path'])
    
    return {
        'model': model,
        'scaler': scaler,
        'time_steps': config['time_steps']
    }

# Plotly Interactive Visualization
def create_plotly_visualization(original_data, test_predict, future_predictions, df, path):
    """
    Create an interactive Plotly visualization of the forecast.
    
    Args:
        original_data (np.array): Original time series data
        test_predict (np.array): Test predictions
        future_predictions (np.array): Future forecast predictions
        df (pd.DataFrame): Original DataFrame
    
    Returns:
        plotly.graph_objs.Figure: Interactive visualization
    """
    # Prepare dates for plotting
    original_dates = df.index
    test_start_date = original_dates[-len(test_predict):]
    future_dates = pd.date_range(start=original_dates[-1], periods=len(future_predictions)+1)[1:]
    
    # Create figure
    fig = go.Figure()
    
    # Original data trace
    fig.add_trace(go.Scatter(
        x=original_dates, 
        y=original_data.flatten(), 
        mode='lines', 
        name='Original Data', 
        line=dict(color='blue')
    ))
    
    # Test predictions trace
    fig.add_trace(go.Scatter(
        x=test_start_date, 
        y=test_predict.flatten(), 
        mode='lines', 
        name='Test Predictions', 
        line=dict(color='red', dash='dot')
    ))
    
    # Future predictions trace
    fig.add_trace(go.Scatter(
        x=future_dates, 
        y=future_predictions.flatten(), 
        mode='lines', 
        name='Future Forecasts', 
        line=dict(color='green', dash='dash')
    ))
    
    # Update layout
    fig.update_layout(
        title=f'{path} Time Series Forecast',
        xaxis_title='Date',
        yaxis_title='Close Price',
        hovermode='x unified',
        legend_title_text='Data Types'
    )
    
    return fig

def forecast_with_lstm(df, column='Close', test_size=0.2, time_steps=10, forecast_days=30, save_model=True, save_dir=""):
    """
    Perform LSTM forecasting on time series data.
    
    Args:
        df (pd.DataFrame): Input DataFrame with time series data
        column (str): Column to forecast
        test_size (float): Proportion of data to use for testing
        time_steps (int): Number of previous time steps to use for prediction
        forecast_days (int): Number of days to forecast
        save_model (bool): Whether to save model artifacts
    
    Returns:
        dict: Forecasting results including predictions, model performance metrics
    """
    # Prepare data
    data = df[column].values.reshape(-1, 1)
    
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Prepare sequences
    X, y = prepare_data(scaled_data, time_steps)
    
    # Split data
    split = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train model
    model = create_lstm_model(input_shape=(time_steps, 1), units=50)
    model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=0)
    
    # Predict and inverse transform
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    train_predict = scaler.inverse_transform(train_predict)
    y_train_inv = scaler.inverse_transform(y_train)
    test_predict = scaler.inverse_transform(test_predict)
    y_test_inv = scaler.inverse_transform(y_test)
    
    # Compute performance metrics
    train_mse = mean_squared_error(y_train_inv, train_predict)
    test_mse = mean_squared_error(y_test_inv, test_predict)
    train_mae = mean_absolute_error(y_train_inv, train_predict)
    test_mae = mean_absolute_error(y_test_inv, test_predict)
    
    # Future forecasting
    last_sequence = scaled_data[-time_steps:].reshape((1, time_steps, 1))
    future_predictions = []
    
    for _ in range(forecast_days):
        next_pred = model.predict(last_sequence)[0]
        future_predictions.append(next_pred[0])
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[0, -1, 0] = next_pred[0]
    
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    
    # Optional model saving
    if save_model:
        save_dir_final = "./trained/" + save_dir.split('/')[-1].split('.')[0]
        save_model_artifacts(model, scaler, time_steps, save_dir_final)

    # Generate and show Plotly figure
    plotly_fig = create_plotly_visualization(data, test_predict, future_predictions, df, save_dir_final)
    plotly_fig.show()
    
    return {
        'future_predictions': future_predictions,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'plotly_figure': plotly_fig
    }

# Example usage demonstrating model saving and loading
def example_model_workflow(path):
    """
    Demonstrate the complete workflow of training, saving, and reusing an LSTM model.
    """
    # Step 1: Load your initial training data    
    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
    
    # Step 2: Train and save the initial model
    print("Training and saving the initial model...")
    initial_results = forecast_with_lstm(
        df, 
        column='Close', 
        test_size=0.2, 
        time_steps=10, 
        forecast_days=30, 
        save_model=True,
        save_dir=path
    )
    
    # Step 3: Later, load the saved model for new predictions
    print("\nLoading saved model artifacts...")
    load_path = path.split('/')[-1].split('.')[0]
    loaded_artifacts = load_model_artifacts(load_path)
    
    # Step 4: Demonstrate how to use the loaded model for prediction
    print("\nUsing loaded model for new predictions...")
    # Prepare new data sequence (must match original training configuration)
    new_data = df['Close'].values[-loaded_artifacts['time_steps']:].reshape(1, -1, 1)
    
    # Normalize the new data using the saved scaler
    new_data_scaled = loaded_artifacts['scaler'].transform(new_data.reshape(-1, 1)).reshape(1, -1, 1)
    
    # Make prediction
    new_prediction_scaled = loaded_artifacts['model'].predict(new_data_scaled)
    new_prediction = loaded_artifacts['scaler'].inverse_transform(new_prediction_scaled)[0][0]
    
    print(f"Next predicted value: {new_prediction}")

# Example running the workflow
if __name__ == "__main__":
    stocks = list_repository_files('../dados/raw')
    
    for stock in stocks:
        example_model_workflow(stock)