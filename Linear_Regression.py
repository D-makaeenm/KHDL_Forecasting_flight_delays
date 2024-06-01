import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Giả sử dữ liệu của bạn được lưu trong một tập tin CSV
flight_data_2023 = pd.read_csv('fl2023_train.csv')

flight_data_2023 = flight_data_2023.drop(columns=['AIRLINE_DOT', 'AIRLINE_CODE', 'DOT_CODE', 'ORIGIN_CITY', 'DEST_CITY', 'CANCELLED', 'CANCELLATION_CODE', 'DIVERTED', 'FL_DATE'])
flight_data_2023 = flight_data_2023.dropna()

# One-hot encoding
categorical_features = ['AIRLINE', 'ORIGIN', 'DEST']
encoder = OneHotEncoder(sparse_output=False)
encoded_features = pd.DataFrame(encoder.fit_transform(flight_data_2023[categorical_features]))
encoded_features.columns = encoder.get_feature_names_out(categorical_features)
encoded_features.index = flight_data_2023.index

# Kết hợp dữ liệu đã mã hóa với các cột khác
data = flight_data_2023.drop(columns=categorical_features)
data = pd.concat([data, encoded_features], axis=1)

# Chuyển tất cả các cột dữ liệu sang kiểu float
data = data.astype(float)

# Chuẩn hóa các cột số
numeric_features = ['CRS_DEP_TIME', 'DEP_TIME', 'DEP_DELAY', 'TAXI_OUT', 'WHEELS_OFF', 'WHEELS_ON', 'TAXI_IN',
                    'CRS_ARR_TIME', 'ARR_TIME', 'CRS_ELAPSED_TIME', 'ELAPSED_TIME', 'AIR_TIME',
                    'DISTANCE', 'DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER', 'DELAY_DUE_NAS', 'DELAY_DUE_SECURITY',
                    'DELAY_DUE_LATE_AIRCRAFT']
scaler = StandardScaler()
data[numeric_features] = scaler.fit_transform(data[numeric_features])

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X = data.drop(columns=['ARR_DELAY'])
y = data['ARR_DELAY']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo và huấn luyện mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Tính toán các chỉ số đánh giá
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
