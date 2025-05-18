🧩 Requirements

Node.js: v20.16.0

📦 Installation

npm install

⚙️ Setup

Tạo file .env trong thư mục gốc của dự án.

Thêm dòng sau (thay YOUR_PYTHON_PATH bằng đường dẫn thực tế đến Python của bạn):

NEXT_PUBLIC_PYTHON=YOUR_PYTHON_PATH

Ví dụ:

NEXT_PUBLIC_PYTHON=/usr/bin/python3

🚀 Run the App

npm run dev

Ứng dụng sẽ chạy tại: http://localhost:3000

📝 How to Use

🔐 Encode (Giấu thông tin)

Truy cập: http://localhost:3000

Chọn file âm thanh.

Nhập văn bản cần giấu.

Đặt tên cho file đầu ra.

Nhấn Submit để thực hiện giấu thông tin.

🔓 Decode (Giải mã thông tin)

Truy cập: http://localhost:3000/decode

Chọn file âm thanh đã được giấu thông tin.

Nhấn Submit để giải mã và xem nội dung.
