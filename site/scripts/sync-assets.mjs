// report/attachment/ を site/public/report-assets/ へコピーする。
// 正本は report/ 側。ここで生成したコピーは git 管理外（site/.gitignore）。
import { cp, mkdir, rm, stat } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const siteRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const repoRoot = path.resolve(siteRoot, '..');
const src = path.join(repoRoot, 'report', 'attachment');
const dest = path.join(siteRoot, 'public', 'report-assets');

async function exists(p) {
  try {
    await stat(p);
    return true;
  } catch {
    return false;
  }
}

if (!(await exists(src))) {
  console.warn(`[sync-assets] ${path.relative(repoRoot, src)} が無いのでスキップします`);
  await mkdir(dest, { recursive: true });
} else {
  await rm(dest, { recursive: true, force: true });
  await mkdir(path.dirname(dest), { recursive: true });
  await cp(src, dest, { recursive: true });
  console.log(`[sync-assets] copied ${path.relative(repoRoot, src)} -> ${path.relative(repoRoot, dest)}`);
}
