import pickle

index_path = "./wad_dataset/frame_index.pkl"

with open(index_path, 'rb') as f:
    frame_index = pickle.load(f)

# 1. Xem danh sách các Folder ID đầu tiên
all_folders = list(frame_index.keys())
print(f"--- Tổng số folder: {len(all_folders)} ---")
print(f"5 folder đầu tiên: {all_folders[:5]}")
print(f"Kiểu dữ liệu của key: {type(all_folders[0])}")

# 2. Xem cấu trúc bên trong 1 folder cụ thể (ví dụ folder đầu tiên)
first_folder = all_folders[0]
print(f"\n--- Chi tiết folder: {first_folder} ---")
all_frames = list(frame_index[first_folder].keys())
print(f"Số lượng frame trong folder này: {len(all_frames)}")
print(f"5 frame ID đầu tiên: {all_frames[:5]}")
print(f"Kiểu dữ liệu của frame_id: {type(all_frames[0])}")

# 3. Xem dữ liệu mẫu của 1 frame
sample_frame = all_frames[0]
print(f"\n--- Dữ liệu mẫu của frame {sample_frame} ---")
print(frame_index[first_folder][sample_frame])