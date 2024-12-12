import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.api import VAR

def fetch_data(ticker, period="10y"):
    """Obtém dados históricos dos ativos usando yfinance."""
    data = yf.download(ticker, period=period)["Adj Close"]
    if data.empty:
        raise ValueError("Nenhum dado foi retornado para os tickers fornecidos.")
    data = data.dropna()  # Remove valores NaN
    data.index = pd.to_datetime(data.index)  # Garantir que o índice seja datetime
    if not data.index.inferred_freq:
        data = data.asfreq('B')  # Define frequência como dias úteis, se não houver
    data = data.ffill()  # Preenche valores ausentes com forward fill
    return data

def lstm_forecast():
    """
    Retorna os valores previstos para o forecast de um ativo usando um modelo LSTM.
    """
    
    return {
        "start_date": "2021-01-01", # Date
        "end_date": "2021-01-10", # Date
        "forecast_results": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109] # Array of forecasted values
    }


def lstm_pre_treinado():
    """
    ! Apenas para fins de demonstração !
    
    Retorna os valores previstos para o forecast de um ativo usando um modelo LSTM que já foi treinado anteriormente
    para apresentar resultados mais rápidos durante a execução do código de demonstração.
    """
    return {
        "start_date": "2021-01-01", # Date
        "end_date": "2021-01-10", # Date
        "forecast_results": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109] # Array of forecasted values
    }
