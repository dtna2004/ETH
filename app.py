import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from data_collector import BinanceDataCollector
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px

class ETHPredictionApp:
    def __init__(self):
        self.collector = BinanceDataCollector()
        self.load_models()
        
    def load_models(self):
        self.rf_model = joblib.load('models/random_forest.joblib')
        self.mlp_model = joblib.load('models/mlp.joblib')
        self.lstm_model = load_model('models/lstm_model.h5')
        self.scaler = joblib.load('models/scaler.joblib')
        
    def prepare_current_features(self):
        data = self.collector.get_historical_data(days=3)
        processed_data = self.collector.prepare_features(data)
        current_features = processed_data[['returns', 'volatility', 'sma_20', 'sma_50', 'rsi']].iloc[-1]
        return current_features, data['close'].iloc[-1]
    
    def predict_probability(self, target_price, timeframe):
        current_features, current_price = self.prepare_current_features()
        features_scaled = self.scaler.transform(current_features.values.reshape(1, -1))
        
        # Dự đoán với từng mô hình
        rf_prob = self.rf_model.predict_proba(features_scaled)[0][1]
        mlp_prob = self.mlp_model.predict_proba(features_scaled)[0][1]
        lstm_prob = self.lstm_model.predict(features_scaled.reshape(1, 1, -1))[0][0]
        
        # Tính trung bình xác suất từ 3 mô hình
        avg_prob = np.mean([rf_prob, mlp_prob, lstm_prob])
        
        # Lưu dự đoán
        prediction_data = {
            'timestamp': datetime.now().isoformat(),
            'current_price': float(current_price),
            'target_price': float(target_price),
            'timeframe': timeframe,
            'probabilities': {
                'random_forest': float(rf_prob),
                'neural_network': float(mlp_prob),
                'lstm': float(lstm_prob),
                'average': float(avg_prob)
            }
        }
        self.collector.save_prediction(prediction_data)
        
        return {
            'Random Forest': rf_prob,
            'Neural Network': mlp_prob,
            'LSTM': lstm_prob,
            'Average': avg_prob
        }
    
    def plot_price_history(self):
        data = self.collector.get_historical_data(days=30)
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=data['timestamp'],
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='ETH/USDT'
        ))
        
        fig.update_layout(
            title='ETH/USDT Price History (30 Days)',
            yaxis_title='Price (USDT)',
            xaxis_title='Date',
            template='plotly_dark',
            height=600
        )
        
        return fig
    
    def plot_accuracy_stats(self):
        stats = self.collector.get_accuracy_stats()
        timeframes = list(stats.keys())
        accuracies = [stats[tf]['accuracy'] for tf in timeframes]
        total_predictions = [stats[tf]['total_predictions'] for tf in timeframes]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=timeframes,
            y=accuracies,
            name='Accuracy (%)',
            marker_color='lightgreen'
        ))
        
        fig.add_trace(go.Bar(
            x=timeframes,
            y=total_predictions,
            name='Total Predictions',
            marker_color='lightblue',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Model Accuracy by Timeframe',
            yaxis=dict(title='Accuracy (%)', side='left', range=[0, 100]),
            yaxis2=dict(title='Total Predictions', side='right', overlaying='y'),
            barmode='group',
            template='plotly_dark',
            height=400
        )
        
        return fig

def main():
    st.set_page_config(
        page_title='ETH Price Prediction',
        layout='wide',
        initial_sidebar_state='expanded'
    )
    
    # Khởi tạo session state nếu chưa có
    if 'target_price' not in st.session_state:
        st.session_state.target_price = None
    if 'percent_change' not in st.session_state:
        st.session_state.percent_change = 0.0
    
    st.title('ETH Price Prediction App 🚀')
    
    app = ETHPredictionApp()
    
    # Sidebar
    st.sidebar.header('Cài đặt dự đoán')
    timeframe = st.sidebar.selectbox(
        'Chọn khung thời gian:',
        ['1h', '4h', '12h', '24h', '3d', '7d']
    )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.plotly_chart(app.plot_price_history(), use_container_width=True)
    
    with col2:
        current_price = float(app.collector.get_historical_data(days=1)['close'].iloc[-1])
        st.metric('Giá ETH hiện tại', f'${current_price:,.2f}')
        
        # Thêm radio button để chọn cách nhập giá mục tiêu
        input_method = st.radio(
            "Chọn cách nhập giá mục tiêu:",
            ["Nhập giá trực tiếp", "Chọn % thay đổi"],
            key='input_method'
        )
        
        if input_method == "Nhập giá trực tiếp":
            # Sử dụng session state để lưu giá trị
            if st.session_state.target_price is None:
                st.session_state.target_price = current_price
            
            price_str = st.text_input(
                'Nhập giá mục tiêu (USDT):',
                value=str(st.session_state.target_price),
                key='price_input'
            )
            
            try:
                target_price = float(price_str)
                st.session_state.target_price = target_price
            except ValueError:
                st.error('Vui lòng nhập một số hợp lệ')
                target_price = st.session_state.target_price
        else:
            # Sử dụng session state cho % thay đổi
            percent_change = st.slider(
                'Chọn % thay đổi giá:',
                min_value=-50.0,
                max_value=50.0,
                value=st.session_state.percent_change,
                step=0.1,
                format='%+.1f%%',
                key='percent_slider'
            )
            st.session_state.percent_change = percent_change
            target_price = current_price * (1 + percent_change/100)
            st.write(f'Giá mục tiêu: ${target_price:,.2f}')
        
        price_change = ((target_price - current_price) / current_price) * 100
        st.write(f'Thay đổi giá: {price_change:+.2f}%')
        
        # Thêm visual feedback về hướng thay đổi giá
        if price_change > 0:
            st.markdown('🔼 Dự đoán tăng giá')
        elif price_change < 0:
            st.markdown('🔽 Dự đoán giảm giá')
        else:
            st.markdown('➡️ Giá không đổi')
    
    if st.button('Dự đoán xác suất', use_container_width=True):
        probabilities = app.predict_probability(target_price, timeframe)
        
        st.subheader('Kết quả dự đoán')
        cols = st.columns(4)
        
        for col, (model, prob) in zip(cols, probabilities.items()):
            with col:
                st.metric(
                    model,
                    f"{prob:.1%}",
                    delta=f"{(prob - 0.5) * 100:+.1f}pp vs. 50%"
                )
    
    # Accuracy stats
    st.subheader('Thống kê độ chính xác')
    st.plotly_chart(app.plot_accuracy_stats(), use_container_width=True)
    
    # Hiển thị lịch sử dự đoán
    st.subheader('Lịch sử dự đoán')
    if app.collector.prediction_history['predictions']:
        df = pd.DataFrame(app.collector.prediction_history['predictions'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Lấy dữ liệu giá realtime một lần
        latest_data = app.collector.get_historical_data(days=7)
        latest_data.set_index('timestamp', inplace=True)
        
        # Thêm cột trạng thái
        def get_status(row):
            if 'verified' in row:
                if row['verified']['is_correct']:
                    return f'✅ Đạt (Cao: ${row["verified"]["period_high"]:,.2f}, Thấp: ${row["verified"]["period_low"]:,.2f})'
                return f'❌ Không đạt (Cao: ${row["verified"]["period_high"]:,.2f}, Thấp: ${row["verified"]["period_low"]:,.2f})'
            
            # Kiểm tra xem đã hết thời gian chưa
            pred_time = pd.to_datetime(row['timestamp'])
            timeframe_delta = {
                '1h': timedelta(minutes=61),
                '4h': timedelta(hours=4, minutes=1),
                '12h': timedelta(hours=12, minutes=1),
                '24h': timedelta(hours=24, minutes=1),
                '3d': timedelta(days=3),
                '7d': timedelta(days=7)
            }
            
            end_time = pred_time + timeframe_delta[row['timeframe']]
            
            # Nếu chưa hết thời gian, hiển thị thời gian còn lại
            if datetime.now() < end_time:
                time_left = end_time - datetime.now()
                hours = int(time_left.total_seconds() // 3600)
                minutes = int((time_left.total_seconds() % 3600) // 60)
                if hours > 0:
                    return f'🕒 Còn {hours}h {minutes}m'
                return f'🕒 Còn {minutes}m'
            
            # Nếu đã hết thời gian, kiểm tra kết quả ngay
            try:
                period_data = latest_data.loc[
                    (latest_data.index >= pred_time) &
                    (latest_data.index <= end_time)
                ]
                
                if not period_data.empty:
                    target_price = row['target_price']
                    current_price = row['current_price']
                    period_high = period_data['high'].max()
                    period_low = period_data['low'].min()
                    
                    # Kiểm tra kết quả
                    if target_price > current_price:
                        is_correct = period_high >= target_price
                    else:
                        is_correct = period_low <= target_price
                        
                    if is_correct:
                        return f'✅ Đạt (Cao: ${period_high:,.2f}, Thấp: ${period_low:,.2f})'
                    return f'❌ Không đạt (Cao: ${period_high:,.2f}, Thấp: ${period_low:,.2f})'
                    
            except Exception as e:
                st.error(f"Lỗi khi kiểm tra kết quả: {str(e)}")
                
            return '⚠️ Lỗi xác nhận'
        
        df['trạng_thái'] = df.apply(get_status, axis=1)
        df = df.sort_values('timestamp', ascending=False)
        
        # Format các cột giá
        df['current_price'] = df['current_price'].apply(lambda x: f'${x:,.2f}')
        df['target_price'] = df['target_price'].apply(lambda x: f'${x:,.2f}')
        
        # Hiển thị với tên cột tiếng Việt
        st.dataframe(
            df.rename(columns={
                'timestamp': 'Thời gian',
                'current_price': 'Giá hiện tại',
                'target_price': 'Giá mục tiêu',
                'timeframe': 'Khung thời gian',
                'trạng_thái': 'Trạng thái'
            })[['Thời gian', 'Giá hiện tại', 'Giá mục tiêu', 'Khung thời gian', 'Trạng thái']],
            use_container_width=True
        )
        
        # Thêm nút làm mới
        if st.button('Làm mới dữ liệu', use_container_width=True):
            st.rerun()
    else:
        st.info('Chưa có dự đoán nào được lưu.')
    
    # Thêm ghi chú
    st.markdown('---')
    st.markdown("""
    **Lưu ý:**
    - Các dự đoán dựa trên dữ liệu lịch sử và không nên được sử dụng như là cơ sở duy nhất cho các quyết định giao dịch
    - Độ chính xác được tính dựa trên các dự đoán trong quá khứ đã được xác nhận
    - Các mô hình được huấn luyện với dữ liệu 1 năm gần nhất
    """)

if __name__ == '__main__':
    main() 