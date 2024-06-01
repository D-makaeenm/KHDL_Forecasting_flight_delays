import pandas as pd

# Tải dữ liệu từ tệp CSV gốc
flight_data_2023 = pd.read_csv("flight_data_2023.csv")

# Tách 20 dòng cuối cùng làm dữ liệu kiểm tra
test_data = flight_data_2023.tail(20)
# Sử dụng phần còn lại của dữ liệu để huấn luyện mô hình
train_data = flight_data_2023.iloc[:-20]

# Lưu 20 dòng cuối cùng vào tệp mới
test_data.to_csv("fl2023_test.csv", index=False)

# Lưu phần dữ liệu huấn luyện vào tệp mới nếu cần
train_data.to_csv("fl2023_train.csv", index=False)
