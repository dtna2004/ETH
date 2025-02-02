# Ứng dụng Dự đoán Xác suất Giá ETH

Ứng dụng này sử dụng dữ liệu từ Binance Futures API và 3 mô hình machine learning khác nhau để dự đoán xác suất giá ETH đạt đến một mức giá mục tiêu.

## Cài đặt

1. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

2. (Tùy chọn) Cấu hình API Binance:
- Đổi tên file `.env.example` thành `.env`
- Thêm API key và secret của Binance vào file `.env`
- Nếu không có API key, ứng dụng vẫn hoạt động nhưng sẽ bị giới hạn số lượng request

3. Huấn luyện các mô hình:
```bash
python train_models.py
```

4. Chạy ứng dụng:
```bash
streamlit run app.py
```

## Cách sử dụng

1. Sau khi chạy ứng dụng, truy cập vào địa chỉ được hiển thị trong terminal (thường là http://localhost:8501)
2. Xem biểu đồ giá ETH trong 30 ngày gần nhất
3. Nhập giá mục tiêu bạn muốn dự đoán xác suất
4. Nhấn nút "Predict Probability" để xem kết quả dự đoán từ 3 mô hình khác nhau

## Các mô hình được sử dụng

1. Random Forest
2. Neural Network (MLP)
3. LSTM (Long Short-Term Memory)

## Lưu ý

- Các dự đoán dựa trên dữ liệu lịch sử và không nên được sử dụng như là cơ sở duy nhất cho các quyết định giao dịch
- Mô hình được huấn luyện với dữ liệu 1 năm gần nhất
- Các features được sử dụng bao gồm: returns, volatility, SMA20, SMA50, và RSI 