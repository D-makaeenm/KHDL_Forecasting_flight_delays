import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.models import load_model
import pickle

# Tải mô hình và các đối tượng encoder, scaler, và tên các cột
model = load_model('flight_delay_model.h5')
with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('X_train_columns.pkl', 'rb') as f:
    X_train_columns = pickle.load(f)

# Định nghĩa lại categorical_features và numeric_features
categorical_features = ['AIRLINE', 'ORIGIN', 'DEST']
numeric_features = ['CRS_DEP_TIME', 'DEP_TIME', 'DEP_DELAY', 'TAXI_OUT', 'WHEELS_OFF', 'WHEELS_ON', 'TAXI_IN',
                    'CRS_ARR_TIME', 'ARR_TIME', 'CRS_ELAPSED_TIME', 'ELAPSED_TIME', 'AIR_TIME',
                    'DISTANCE', 'DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER', 'DELAY_DUE_NAS', 'DELAY_DUE_SECURITY',
                    'DELAY_DUE_LATE_AIRCRAFT']

# Tạo một số dữ liệu mới cho dữ liệu test
test_data_new = pd.DataFrame({
    'AIRLINE': ['Delta Air Lines Inc.', 'American Airlines Inc.'],
    'ORIGIN': ['MSP', 'LAX'],
    'DEST': ['SFO', 'JFK'],
    'CRS_DEP_TIME': [12.0, 15.0],
    'DISTANCE': [1500, 2500],
    'CRS_ELAPSED_TIME': [240, 300],
    'DEP_TIME': [10, 13],
    'DEP_DELAY': [2, 3],
    'TAXI_OUT': [20, 25],
    'WHEELS_OFF': [30, 38],
    'WHEELS_ON': [45, 50],
    'TAXI_IN': [8, 10],
    'CRS_ARR_TIME': [50, 55],
    'ARR_TIME': [53, 58],
    'ELAPSED_TIME': [263, 305],
    'AIR_TIME': [235, 280],
    'DELAY_DUE_CARRIER': [5, 7],
    'DELAY_DUE_WEATHER': [0, 0],
    'DELAY_DUE_NAS': [0, 0],
    'DELAY_DUE_SECURITY': [0, 0],
    'DELAY_DUE_LATE_AIRCRAFT': [0, 0]
})

# Thêm cột 'FL_NUMBER' với giá trị mặc định 0 (hoặc giá trị phù hợp)
test_data_new['FL_NUMBER'] = 0

# Áp dụng mã hóa one-hot cho dữ liệu thử nghiệm mới
test_encoded_features = pd.DataFrame(encoder.transform(test_data_new[categorical_features]))
test_encoded_features.columns = encoder.get_feature_names_out(categorical_features)

# Kết hợp dữ liệu thử nghiệm mới với các cột số đã chuẩn hóa
test_data_new = test_data_new.drop(columns=categorical_features)
test_data_new = pd.concat([test_data_new, test_encoded_features], axis=1)

# Chuẩn hóa các đặc trưng số trong dữ liệu thử nghiệm mới
test_data_new[numeric_features] = scaler.transform(test_data_new[numeric_features])

# Đảm bảo dữ liệu thử nghiệm mới có cùng các đặc trưng và thứ tự với X_train
test_data_new = test_data_new[X_train_columns]

# Dự đoán độ trễ của chuyến bay với mô hình đã huấn luyện
predicted_delay_new = model.predict(test_data_new)

# In kết quả dự đoán
print("Dự đoán độ trễ của các chuyến bay mới:")
for i in range(len(predicted_delay_new)):
    print(f"Chuyến bay {i+1}: {predicted_delay_new[i][0]} phút")
