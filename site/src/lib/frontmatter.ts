/**
 * 依存を増やさないための最小 YAML frontmatter パーサ。
 * `key: value` と `key:` + `  - item` の配列のみを扱う。
 * 解釈できない行は無視する（frontmatter が原因でビルドを落とさない）。
 */
export type Frontmatter = Record<string, string | string[]>;

const FM_RE = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/;

function unquote(value: string): string {
  const trimmed = value.trim();
  if (
    (trimmed.startsWith('"') && trimmed.endsWith('"') && trimmed.length >= 2) ||
    (trimmed.startsWith("'") && trimmed.endsWith("'") && trimmed.length >= 2)
  ) {
    return trimmed.slice(1, -1);
  }
  return trimmed;
}

export function splitFrontmatter(raw: string): { data: Frontmatter; body: string; warning?: string } {
  const match = raw.match(FM_RE);
  if (!match) return { data: {}, body: raw };

  const data: Frontmatter = {};
  let currentKey: string | null = null;
  let warning: string | undefined;

  for (const line of match[1].split(/\r?\n/)) {
    if (!line.trim() || line.trim().startsWith('#')) continue;

    const listItem = line.match(/^\s+-\s+(.*)$/);
    if (listItem && currentKey) {
      const existing = data[currentKey];
      const list = Array.isArray(existing) ? existing : [];
      list.push(unquote(listItem[1]));
      data[currentKey] = list;
      continue;
    }

    const pair = line.match(/^([A-Za-z0-9_-]+)\s*:\s*(.*)$/);
    if (pair) {
      const [, key, value] = pair;
      currentKey = key;
      if (value.trim() === '') {
        data[key] = [];
      } else {
        data[key] = unquote(value);
      }
      continue;
    }

    warning = `frontmatter の解釈できない行: ${line.trim()}`;
  }

  return { data, body: raw.slice(match[0].length), warning };
}

export function fmString(data: Frontmatter, key: string): string | undefined {
  const value = data[key];
  if (typeof value === 'string' && value.trim() !== '') return value.trim();
  return undefined;
}

export function fmNumber(data: Frontmatter, key: string): number | undefined {
  const value = fmString(data, key);
  if (value === undefined) return undefined;
  const n = Number(value);
  return Number.isFinite(n) ? n : undefined;
}

export function fmList(data: Frontmatter, key: string): string[] {
  const value = data[key];
  if (Array.isArray(value)) return value.filter((v) => v.trim() !== '');
  if (typeof value === 'string' && value.trim() !== '') {
    return value
      .split(',')
      .map((v) => v.trim())
      .filter(Boolean);
  }
  return [];
}
