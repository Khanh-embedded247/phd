import pickle
from pprint import pprint

file_path = "/home/khanh247/Documents/Survey/phd/phenikaa/data/nuscenes/raw/nuscenes_map_infos_temporal_train.pkl"

with open(file_path, "rb") as f:
    data = pickle.load(f)

print(f"--- KIỂU DỮ LIỆU GỐC: {type(data)} ---")

if isinstance(data, dict):
    # Lấy ra tối đa 100 keys đầu tiên để xem cấu trúc
    first_100_keys = list(data.keys())[:100]
    print(f"\n[Hiển thị {len(first_100_keys)} phần tử đầu tiên của Dictionary]:")
    for key in first_100_keys:
        # In ra tên key và kiểu dữ liệu của giá trị bên trong key đó
        value_info = type(data[key])
        if hasattr(data[key], 'shape'):  # Nếu là mảng numpy/tensor thì in thêm kích thước
            value_info = f"{value_info} | Shape: {data[key].shape}"
        print(f"🔑 Key: {key} -> {value_info}")
else:
    # Nếu là chuỗi văn bản hoặc danh sách phẳng, ép kiểu chuỗi và cắt 100 dòng đầu
    lines = str(data).split('\n')
    print('\n'.join(lines[:100]))
