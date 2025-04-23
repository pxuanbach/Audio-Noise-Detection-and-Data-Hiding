'use client';

import React, { useState } from 'react';

export default function Home() {
  const [fileInfo, setFileInfo] = useState<File | null>(null);
  const [textInput, setTextInput] = useState('');
  const [newFileName, setNewFileName] = useState('');
  const [downloadUrl, setDownloadUrl] = useState('');

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setFileInfo(file);
      setDownloadUrl('');
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!fileInfo) {
      alert('Vui lòng chọn file âm thanh');
      return;
    }
    if (!newFileName) {
      alert('Vui lòng nhập tên file mới');
      return;
    }

    const formData = new FormData();
    formData.append('file', fileInfo);
    formData.append('text', textInput);
    formData.append('newFileName', newFileName);

    try {
      const res = await fetch('/api/encode', {
        method: 'POST',
        body: formData,
      });

      const result = await res.json();

      if (result.downloadUrl) {
        setDownloadUrl(result.downloadUrl);
      } else {
        alert(result.error || 'Đã xảy ra lỗi khi xử lý file');
      }
    } catch (err) {
      alert('Lỗi kết nối tới server');
      console.error(err);
    }
  };

  return (
    <main className="min-h-screen flex flex-col items-center justify-center bg-black-100 p-4">
      <form
        onSubmit={handleSubmit}
        className="bg-black p-6 rounded shadow-md w-full max-w-md"
      >
        <h1 className="text-2xl font-bold mb-4">Tải lên file âm thanh</h1>

        <input
          type="file"
          accept="audio/*"
          onChange={handleFileChange}
          className="mb-4 block w-full"
        />

        <input
          type="text"
          placeholder="Nhập ghi chú / text"
          value={textInput}
          onChange={(e) => setTextInput(e.target.value)}
          className="mb-4 w-full px-3 py-2 border rounded"
        />

        <input
          type="text"
          placeholder="Nhập tên file mới (vd: file_moi.wav)"
          value={newFileName}
          onChange={(e) => setNewFileName(e.target.value)}
          className="mb-4 w-full px-3 py-2 border rounded"
        />

        <button
          type="submit"
          className="w-full bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700"
        >
          Gửi tới Python script
        </button>
      </form>

      {fileInfo && (
        <div className="mt-6 bg-black shadow-md rounded p-4 w-full max-w-md">
          <p><strong>Tên file:</strong> {fileInfo.name}</p>
          <p><strong>Kích thước:</strong> {(fileInfo.size / 1024).toFixed(2)} KB</p>
          <p><strong>Định dạng:</strong> {fileInfo.type}</p>
        </div>
      )}

      {downloadUrl && (
        <a
          href={downloadUrl}
          download
          className="mt-6 inline-block px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
        >
          ⬇️ Tải file kết quả
        </a>
      )}
    </main>
  );
}
