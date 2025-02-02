# Ứng dụng Dự đoán Xác suất Giá ETH

Ứng dụng này sử dụng dữ liệu từ Binance Futures API và 3 mô hình machine learning khác nhau để dự đoán xác suất giá ETH đạt đến một mức giá mục tiêu trong các khung thời gian khác nhau.

## Tính năng

- Dự đoán xác suất giá ETH đạt mục tiêu với 3 mô hình:
  - Random Forest
  - Neural Network (MLP)
  - LSTM (Long Short-Term Memory)
- Hỗ trợ nhiều khung thời gian:
  - 1 giờ
  - 4 giờ
  - 12 giờ
  - 24 giờ
  - 3 ngày
  - 7 ngày
- Theo dõi và hiển thị độ chính xác của các dự đoán
- Biểu đồ giá realtime
- Giao diện web thân thiện với người dùng

## Cài đặt

1. Clone repository:
```bash
git clone https://github.com/your-username/eth-price-prediction.git
cd eth-price-prediction
```

2. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

3. Cấu hình API Binance (tùy chọn):
- Copy file `.env.example` thành `.env`
- Thêm API key và secret của Binance vào file `.env`
- Nếu không có API key, ứng dụng vẫn hoạt động nhưng sẽ bị giới hạn số lượng request

4. Huấn luyện các mô hình:
```bash
python train_models.py
```

5. Chạy ứng dụng:
```bash
streamlit run app.py
```

## Cập nhật độ chính xác

Để cập nhật độ chính xác của các dự đoán, chạy:
```bash
python update_accuracy.py
```

Bạn có thể thiết lập một task tự động chạy script này mỗi giờ để cập nhật kết quả dự đoán.

## Lưu ý

- Các dự đoán dựa trên dữ liệu lịch sử và không nên được sử dụng như là cơ sở duy nhất cho các quyết định giao dịch
- Độ chính xác được tính dựa trên các dự đoán trong quá khứ đã được xác nhận
- Các mô hình được huấn luyện với dữ liệu 1 năm gần nhất
- Các features được sử dụng bao gồm: returns, volatility, SMA20, SMA50, và RSI

## Đóng góp

Mọi đóng góp đều được hoan nghênh! Vui lòng tạo issue hoặc pull request nếu bạn muốn cải thiện ứng dụng. 