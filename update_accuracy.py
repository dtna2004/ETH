from data_collector import BinanceDataCollector
from datetime import datetime, timedelta
import json
from pathlib import Path
import pandas as pd

def get_timeframe_delta(timeframe):
    if timeframe == '1h':
        return timedelta(hours=1)
    elif timeframe == '4h':
        return timedelta(hours=4)
    elif timeframe == '12h':
        return timedelta(hours=12)
    elif timeframe == '24h':
        return timedelta(hours=24)
    elif timeframe == '3d':
        return timedelta(days=3)
    elif timeframe == '7d':
        return timedelta(days=7)
    return None

def check_price_target_reached(data, target_price, current_price):
    """Kiểm tra xem giá mục tiêu có đạt được trong khoảng thời gian không."""
    if target_price > current_price:
        # Dự đoán tăng giá
        return data['high'].max() >= target_price
    else:
        # Dự đoán giảm giá
        return data['low'].min() <= target_price

def update_prediction_accuracy():
    collector = BinanceDataCollector()
    predictions_dir = Path('predictions')
    history_file = predictions_dir / 'prediction_history.json'
    
    if not history_file.exists():
        print("No prediction history found.")
        return
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    current_time = datetime.now()
    updated_predictions = []
    updated_count = 0
    
    # Lấy dữ liệu giá một lần cho tất cả các dự đoán
    latest_data = collector.get_historical_data(days=7)
    latest_data.set_index('timestamp', inplace=True)
    
    for pred in history['predictions']:
        # Bỏ qua các dự đoán đã được xác nhận
        if 'verified' in pred:
            updated_predictions.append(pred)
            continue
        
        pred_time = pd.to_datetime(pred['timestamp'])
        timeframe_delta = get_timeframe_delta(pred['timeframe'])
        end_time = pred_time + timeframe_delta
        
        # Kiểm tra xem đã hết thời gian dự đoán chưa
        if current_time >= end_time:
            try:
                # Lấy tất cả dữ liệu giá trong khoảng thời gian dự đoán
                period_data = latest_data.loc[
                    (latest_data.index >= pred_time) &
                    (latest_data.index <= end_time)
                ]
                
                if not period_data.empty:
                    target_price = pred['target_price']
                    current_price = pred['current_price']
                    
                    # Kiểm tra xem có đạt giá mục tiêu không
                    is_correct = check_price_target_reached(period_data, target_price, current_price)
                    
                    # Lấy giá cao nhất và thấp nhất trong khoảng thời gian
                    period_high = period_data['high'].max()
                    period_low = period_data['low'].min()
                    final_price = period_data['close'].iloc[-1]
                    
                    # Cập nhật thống kê
                    collector.update_accuracy(pred['timeframe'], is_correct)
                    
                    # Thêm thông tin xác nhận vào dự đoán
                    pred['verified'] = {
                        'is_correct': is_correct,
                        'verified_at': current_time.isoformat(),
                        'final_price': float(final_price),
                        'period_high': float(period_high),
                        'period_low': float(period_low)
                    }
                    updated_count += 1
                    
            except (IndexError, KeyError) as e:
                print(f"Error processing prediction from {pred_time}: {str(e)}")
                pass
            
        updated_predictions.append(pred)
    
    # Cập nhật lịch sử dự đoán
    history['predictions'] = updated_predictions
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"Updated {updated_count} predictions.")

if __name__ == '__main__':
    update_prediction_accuracy() 