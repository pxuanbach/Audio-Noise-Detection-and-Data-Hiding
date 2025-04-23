/* eslint-disable @typescript-eslint/no-explicit-any */
import { NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs/promises';

export async function POST(req: Request) {
  try {
    const formData = await req.formData();
    const file = formData.get('file') as File;
    const newFileName = formData.get('newFileName') as string | null;
    const text = formData.get('text') as string | null; // có thể dùng sau nếu muốn

    if (!file) {
      return NextResponse.json({ error: 'Thiếu file tải lên' }, { status: 400 });
    }

    const uploadsDir = path.join(process.cwd(), 'public', 'uploads');
    await fs.mkdir(uploadsDir, { recursive: true });

    const buffer = Buffer.from(await file.arrayBuffer());
    const originalFilename = file.name;
    const inputPath = path.join(uploadsDir, originalFilename);
    await fs.writeFile(inputPath, buffer);

    // Tên file đầu ra
    const finalOutputName = newFileName?.trim() || `processed_${originalFilename}`;
    const outputPath = path.join(uploadsDir, finalOutputName);

    const inputText = text ? text : "";

    // Gọi script Python
    return await new Promise((resolve) => {
      const py = spawn(
        'C:/Python312/python.exe', // hoặc 'python' tùy môi trường của bạn
        [path.join(process.cwd(), 'scripts', 'encode_script.py'), inputPath, outputPath, inputText],
        {
          env: {
            ...process.env,
            PYTHONIOENCODING: 'utf-8',
          },
        }
      );

      let errorOutput = '';

      py.stderr.on('data', (data) => {
        errorOutput += data.toString();
      });

      py.on('close', (code) => {
        if (code === 0) {
          const downloadUrl = `/uploads/${finalOutputName}`;
          resolve(NextResponse.json({ downloadUrl }));
        } else {
          resolve(
            NextResponse.json(
              { error: `Python script lỗi:\n${errorOutput}` },
              { status: 500 }
            )
          );
        }
      });
    });
  } catch (err: any) {
    return NextResponse.json({ error: err.message }, { status: 500 });
  }
}
