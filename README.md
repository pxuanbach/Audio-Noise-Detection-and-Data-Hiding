# Audio Steganography Web App

## 🧩 Requirements

- **Node.js**: `v20.16.0`

## 📦 Installation

```bash
npm install
```

## ⚙️ Setup

1. Tạo file `.env` trong thư mục gốc của dự án.
2. Thêm dòng sau (thay `YOUR_PYTHON_PATH` bằng đường dẫn thực tế đến Python của bạn):

   ```env
   NEXT_PUBLIC_PYTHON=YOUR_PYTHON_PATH
   ```

   **Ví dụ:**

   ```env
   NEXT_PUBLIC_PYTHON=/usr/bin/python3
   ```

## 🚀 Run the App

```bash
npm run dev
```

Ứng dụng sẽ chạy tại: [http://localhost:3000](http://localhost:3000)

---

## 📝 How to Use

### 🔐 Encode (Giấu thông tin)

1. Truy cập: [http://localhost:3000](http://localhost:3000)
2. Chọn file âm thanh.
3. Nhập văn bản cần giấu.
4. Đặt tên cho file đầu ra.
5. Nhấn **Submit** để thực hiện giấu thông tin.

### 🔓 Decode (Giải mã thông tin)

1. Truy cập: [http://localhost:3000/decode](http://localhost:3000/decode)
2. Chọn file âm thanh đã được giấu thông tin.
3. Nhấn **Submit** để giải mã và xem nội dung.
