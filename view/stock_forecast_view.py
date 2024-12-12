from fastapi import APIRouter, HTTPException
import pandas as pd

from model.stock_forecast_model import StockForecast
from controller.forecast_var_controller import fetch_data

stock_forecast  = APIRouter()

@stock_forecast.post("/stock_forecast/")
async def stock_forecast_lstm(stock_forecast: StockForecast):
    try:
        # Obter os dados históricos
        data = fetch_data(stock_forecast.tickers)
        
        # Chamada para modelo lstm
        # returned_data = lstm_forecast(data, days_to_forecast)

        return {
            start_date: returned_data.start_date, # Dia inicial do período de previsão
            end_date: returned_data.end_date, # Dia final do período de previsão
            forecast_by_day: returned_data.forecast_results # Array de valores previstos por dia
        }
    except Exception as e:
        import traceback
        traceback.print_exc()  # Exibe o rastreamento completo no log do console
        raise HTTPException(status_code=400, detail=str(e))
