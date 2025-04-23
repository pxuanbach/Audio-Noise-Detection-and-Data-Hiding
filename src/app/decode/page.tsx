/* eslint-disable @typescript-eslint/no-explicit-any */
'use client'

import { useState } from 'react';

export default function PythonResultPage() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<string>('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    setLoading(true);
    try {
      const res = await fetch('/api/decode', {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();
      if (data.result) setResult(data.result);
      else setResult(data.error || 'Unknown error');
    } catch (err: any) {
      setResult('Có lỗi xảy ra!');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-xl mx-auto mt-10 p-4 border rounded shadow">
      <h1 className="text-2xl font-bold mb-4">Chạy Python Script với âm thanh</h1>
      <form onSubmit={handleSubmit} className="space-y-4">
        <input
          type="file"
          accept="audio/*"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
          className="block w-full border p-2 rounded"
        />
        <button
          type="submit"
          className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded"
        >
          {loading ? 'Đang xử lý...' : 'Gửi đến Python'}
        </button>
      </form>

      {result && (
        <div className="mt-6 p-4 bg-black-100 rounded">
          <h2 className="font-semibold">Kết quả từ script:</h2>
          <pre className="mt-2 text-sm">{result}</pre>
        </div>
      )}
    </div>
  );
}
