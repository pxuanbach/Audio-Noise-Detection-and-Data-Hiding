# scripts/my_script.py
import sys
import shutil

input_path = sys.argv[1]
output_path = sys.argv[2]

# Giả sử xử lý: chỉ copy sang file khác tên
shutil.copyfile(input_path, output_path)
print(f"Đã tạo file mới tại {output_path}")
