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
        
        target_price = st.number_input(
            'Nhập giá mục tiêu (USDT):',
            min_value=0.0,
            value=current_price * 1.1,
            step=100.0,
            format='%.2f'
        )
        
        price_change = ((target_price - current_price) / current_price) * 100
        st.write(f'Thay đổi giá: {price_change:+.2f}%')
    
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
        df = df.sort_values('timestamp', ascending=False)
        
        st.dataframe(
            df[['timestamp', 'current_price', 'target_price', 'timeframe']],
            use_container_width=True
        )
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