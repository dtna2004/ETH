from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np
import joblib
from data_collector import BinanceDataCollector

class ModelTrainer:
    def __init__(self):
        self.collector = BinanceDataCollector()
        self.scaler = StandardScaler()
        
    def prepare_data(self):
        # Lấy và xử lý dữ liệu
        data = self.collector.get_historical_data(days=365)
        processed_data = self.collector.prepare_features(data)
        
        # Chuẩn bị features và target
        features = ['returns', 'volatility', 'sma_20', 'sma_50', 'rsi']
        X = processed_data[features]
        y = processed_data['target']
        
        # Chuẩn hóa dữ liệu
        X_scaled = self.scaler.fit_transform(X)
        
        # Chia dữ liệu
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test, X_scaled.shape[1]
    
    def train_random_forest(self, X_train, y_train):
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        joblib.dump(rf_model, 'models/random_forest.joblib')
        return rf_model
    
    def train_mlp(self, X_train, y_train):
        mlp_model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=1000,
            random_state=42
        )
        mlp_model.fit(X_train, y_train)
        joblib.dump(mlp_model, 'models/mlp.joblib')
        return mlp_model
    
    def train_lstm(self, X_train, y_train, input_dim):
        # Reshape data for LSTM [samples, timesteps, features]
        X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        
        model = Sequential([
            LSTM(50, input_shape=(1, input_dim), return_sequences=True),
            Dropout(0.2),
            LSTM(30),
            Dense(20, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        model.fit(
            X_train_reshaped,
            y_train,
            epochs=50,
            batch_size=32,
            verbose=1
        )
        
        model.save('models/lstm_model.h5')
        return model
    
    def train_all_models(self):
        import os
        if not os.path.exists('models'):
            os.makedirs('models')
            
        X_train, X_test, y_train, y_test, input_dim = self.prepare_data()
        
        # Huấn luyện các mô hình
        rf_model = self.train_random_forest(X_train, y_train)
        mlp_model = self.train_mlp(X_train, y_train)
        lstm_model = self.train_lstm(X_train, y_train, input_dim)
        
        # Lưu scaler
        joblib.dump(self.scaler, 'models/scaler.joblib')
        
        return rf_model, mlp_model, lstm_model

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_all_models() 