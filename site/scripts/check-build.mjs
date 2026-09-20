// 生成物の最低限の検査:
//  - dist/index.html と dist/reports/ がある
//  - report/*.md と同じ本数のレポートページが生成されている
//  - ページ内のローカルリンク・画像が dist に実在する
import { readdir, readFile, stat } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const siteRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const repoRoot = path.resolve(siteRoot, '..');
const dist = path.join(siteRoot, 'dist');
const base = '/bench-of-us/';

const errors = [];

async function exists(p) {
  try {
    await stat(p);
    return true;
  } catch {
    return false;
  }
}

async function walk(dir) {
  const out = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...(await walk(full)));
    else if (entry.name.endsWith('.html')) out.push(full);
  }
  return out;
}

if (!(await exists(path.join(dist, 'index.html')))) errors.push('dist/index.html がありません');
if (!(await exists(path.join(dist, 'reports')))) errors.push('dist/reports/ がありません');

const reportFiles = (await readdir(path.join(repoRoot, 'report'))).filter((f) => f.endsWith('.md'));
if (reportFiles.length < 2) errors.push(`report/*.md が ${reportFiles.length} 本しかありません（2 本以上を期待）`);

for (const file of reportFiles) {
  const slug = file.replace(/\.md$/, '');
  if (!(await exists(path.join(dist, 'reports', slug, 'index.html')))) {
    errors.push(`レポートページが生成されていません: reports/${slug}/`);
  }
}

const pages = await walk(dist);
let checked = 0;

for (const page of pages) {
  const html = await readFile(page, 'utf8');
  for (const match of html.matchAll(/(?:href|src)="([^"]+)"/g)) {
    let url = match[1];
    if (/^(https?:|mailto:|#|data:)/.test(url)) continue;
    url = url.split('#')[0].split('?')[0];
    if (!url) continue;
    if (!url.startsWith(base)) {
      errors.push(`${path.relative(dist, page)}: base 付きでないローカル URL: ${url}`);
      continue;
    }
    const rel = url.slice(base.length);
    const target = url.endsWith('/') ? path.join(dist, rel, 'index.html') : path.join(dist, rel);
    checked += 1;
    if (!(await exists(target))) errors.push(`${path.relative(dist, page)}: リンク切れ: ${url}`);
  }
}

console.log(
  `[check-build] pages=${pages.length} reports=${reportFiles.length} local-links=${checked} errors=${errors.length}`,
);

if (errors.length > 0) {
  for (const error of errors) console.error(`[check-build] ${error}`);
  process.exit(1);
}
