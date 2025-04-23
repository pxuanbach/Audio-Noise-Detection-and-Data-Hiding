import { NextRequest, NextResponse } from 'next/server';
import path from 'path';
import fs from 'fs/promises';
import { spawn } from 'child_process';
import { v4 as uuidv4 } from 'uuid';
import os from 'os';

export async function POST(req: NextRequest) {
  const formData = await req.formData();
  const file = formData.get('file') as File;

  if (!file) {
    return NextResponse.json({ error: 'Không có file được gửi lên.' }, { status: 400 });
  }

  const bytes = await file.arrayBuffer();
  const buffer = Buffer.from(bytes);
  const tempDir = path.join(os.tmpdir(), 'uploads');
  await fs.mkdir(tempDir, { recursive: true });

  const inputPath = path.join(tempDir, `${uuidv4()}.wav`);
  await fs.writeFile(inputPath, buffer);

  return await new Promise((resolve) => {
    const py = spawn(
      'python',
      [path.join(process.cwd(), 'scripts', 'decode_script.py'), inputPath],
      {
        env: {
          ...process.env,
          PYTHONIOENCODING: 'utf-8',
        },
        shell: true,
      }
    );

    let result = '';
    let errorOutput = '';

    py.stdout.on('data', (data) => {
      result += data.toString();
    });

    py.stderr.on('data', (data) => {
      errorOutput += data.toString();
    });

    py.on('close', (code) => {
      if (code === 0) {
        resolve(NextResponse.json({ result: result.trim() }));
      } else {
        resolve(
          NextResponse.json({ error: errorOutput || 'Lỗi không xác định từ Python.' }, { status: 500 })
        );
      }
    });
  });
}
