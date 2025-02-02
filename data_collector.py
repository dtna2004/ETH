from binance.client import Client
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import asyncio
import nest_asyncio
import json
from pathlib import Path

# Cho phép chạy asyncio trong Jupyter/IPython
nest_asyncio.apply()

class BinanceDataCollector:
    def __init__(self):
        # Tải API key và secret từ biến môi trường
        load_dotenv()
        api_key = os.getenv('BINANCE_API_KEY', '')
        api_secret = os.getenv('BINANCE_API_SECRET', '')
        
        # Khởi tạo event loop cho asyncio
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())
        
        # Khởi tạo client với hoặc không có API key
        if api_key and api_secret:
            self.client = Client(api_key, api_secret)
        else:
            self.client = Client()
            print("Warning: Running in public API mode with rate limits. Add API keys for better performance.")
        
        self.symbol = 'ETHUSDT'
        
        # Tạo thư mục predictions nếu chưa tồn tại
        self.predictions_dir = Path('predictions')
        self.predictions_dir.mkdir(exist_ok=True)
        
        # Load lịch sử dự đoán
        self.prediction_history = self.load_prediction_history()

    def get_historical_data(self, days=30, start_date=None, end_date=None):
        if start_date and end_date:
            start_time = start_date
            end_time = end_date
        else:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
        
        klines = self.client.get_historical_klines(
            self.symbol,
            Client.KLINE_INTERVAL_1HOUR,
            str(start_time),
            str(end_time)
        )
        
        # Chuyển đổi dữ liệu thành DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 
            'volume', 'close_time', 'quote_asset_volume',
            'number_of_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        
        # Xử lý dữ liệu
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        
        return df

    def prepare_features(self, df):
        # Tạo các features cho mô hình
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=24).std()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['rsi'] = self.calculate_rsi(df['close'])
        
        # Thêm target (1 nếu giá tăng trong 24h tiếp theo, 0 nếu giảm)
        df['target'] = (df['close'].shift(-24) > df['close']).astype(int)
        
        return df.dropna()

    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def load_prediction_history(self):
        history_file = self.predictions_dir / 'prediction_history.json'
        if history_file.exists():
            with open(history_file, 'r') as f:
                return json.load(f)
        return {
            'predictions': [],
            'accuracy_by_timeframe': {
                '1h': {'correct': 0, 'total': 0},
                '4h': {'correct': 0, 'total': 0},
                '12h': {'correct': 0, 'total': 0},
                '24h': {'correct': 0, 'total': 0},
                '3d': {'correct': 0, 'total': 0},
                '7d': {'correct': 0, 'total': 0}
            }
        }

    def save_prediction(self, prediction_data):
        self.prediction_history['predictions'].append(prediction_data)
        history_file = self.predictions_dir / 'prediction_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.prediction_history, f, indent=2)

    def update_accuracy(self, timeframe, is_correct):
        if timeframe in self.prediction_history['accuracy_by_timeframe']:
            self.prediction_history['accuracy_by_timeframe'][timeframe]['total'] += 1
            if is_correct:
                self.prediction_history['accuracy_by_timeframe'][timeframe]['correct'] += 1
            
            # Lưu lại kết quả
            history_file = self.predictions_dir / 'prediction_history.json'
            with open(history_file, 'w') as f:
                json.dump(self.prediction_history, f, indent=2)

    def get_accuracy_stats(self):
        stats = {}
        for timeframe, data in self.prediction_history['accuracy_by_timeframe'].items():
            if data['total'] > 0:
                accuracy = (data['correct'] / data['total']) * 100
                stats[timeframe] = {
                    'accuracy': accuracy,
                    'total_predictions': data['total'],
                    'correct_predictions': data['correct']
                }
            else:
                stats[timeframe] = {
                    'accuracy': 0,
                    'total_predictions': 0,
                    'correct_predictions': 0
                }
        return stats

if __name__ == "__main__":
    collector = BinanceDataCollector()
    data = collector.get_historical_data(days=60)
    processed_data = collector.prepare_features(data)
    print(processed_data.head()) 