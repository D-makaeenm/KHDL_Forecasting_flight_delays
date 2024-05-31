import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pickle

# Load dữ liệu
flight_data = pd.read_csv("flights_sample_3m.csv")

# Lọc dữ liệu năm 2023
flight_data_2023 = flight_data[flight_data['FL_DATE'].str.startswith('2023')]

# Loại bỏ các cột không cần thiết
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

# Tách dữ liệu thành tập huấn luyện và kiểm tra
X = data.drop(columns=['ARR_DELAY'])
y = data['ARR_DELAY']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây dựng mô hình ANN
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)
])

# Biên dịch mô hình
model.compile(optimizer='adam', loss='mse')

# Huấn luyện mô hình
model.fit(X_train, y_train, epochs=20, validation_split=0.2)

# Lưu mô hình và các đối tượng encoder, scaler
model.save('flight_delay_model.h5')
with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Lưu tên các cột trong X_train
X_train_columns = X_train.columns.tolist()
with open('X_train_columns.pkl', 'wb') as f:
    pickle.dump(X_train_columns, f)

print("Model training and saving completed.")
