from pydantic import BaseModel
from typing import List

class StockForecast(BaseModel):
    """StockForecast model
    
    Attributes:
    tickers: str - Stock ticker to forecast. Ex: "AAPL", "GOOGL", "PETR4.SA"
    """
    ticker: str 
    days_to_forecast: int