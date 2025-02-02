from data_collector import BinanceDataCollector
from datetime import datetime, timedelta
import json
from pathlib import Path

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
    
    for pred in history['predictions']:
        # Bỏ qua các dự đoán đã được xác nhận
        if 'verified' in pred:
            updated_predictions.append(pred)
            continue
        
        pred_time = datetime.fromisoformat(pred['timestamp'])
        timeframe_delta = get_timeframe_delta(pred['timeframe'])
        
        # Kiểm tra xem đã đến thời gian xác nhận chưa
        if current_time >= pred_time + timeframe_delta:
            # Lấy giá thực tế tại thời điểm kết thúc
            end_time = pred_time + timeframe_delta
            historical_data = collector.get_historical_data(
                start_date=end_time - timedelta(hours=1),
                end_date=end_time + timedelta(hours=1)
            )
            
            if not historical_data.empty:
                actual_price = historical_data['close'].iloc[-1]
                target_price = pred['target_price']
                current_price = pred['current_price']
                
                # Xác định dự đoán có chính xác không
                if target_price > current_price:
                    is_correct = actual_price >= target_price
                else:
                    is_correct = actual_price <= target_price
                
                # Cập nhật thống kê
                collector.update_accuracy(pred['timeframe'], is_correct)
                
                # Thêm thông tin xác nhận vào dự đoán
                pred['verified'] = {
                    'actual_price': float(actual_price),
                    'is_correct': is_correct,
                    'verified_at': current_time.isoformat()
                }
            
        updated_predictions.append(pred)
    
    # Cập nhật lịch sử dự đoán
    history['predictions'] = updated_predictions
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)

if __name__ == '__main__':
    update_prediction_accuracy() 