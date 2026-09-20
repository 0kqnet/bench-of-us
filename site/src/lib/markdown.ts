import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkRehype from 'remark-rehype';
import rehypeSanitize, { defaultSchema } from 'rehype-sanitize';
import rehypeStringify from 'rehype-stringify';

type HastNode = {
  type: string;
  tagName?: string;
  properties?: Record<string, unknown>;
  children?: HastNode[];
  value?: string;
};

export type RewriteOptions = {
  /** サイトの base（例: `/bench-of-us`）。末尾スラッシュなし。 */
  base: string;
  /** `report/attachment/<slug>/…` に対応する slug。 */
  slug: string;
};

const ABSOLUTE = /^([a-z][a-z0-9+.-]*:|\/\/|#|\/)/i;

function rewriteUrl(url: string, { base, slug }: RewriteOptions): string {
  if (!url || ABSOLUTE.test(url)) return url;

  const clean = url.replace(/^\.\//, '');

  // 添付（画像・JSON など）は public/report-assets/ にコピーされている
  if (clean.startsWith('attachment/')) {
    return `${base}/report-assets/${clean.slice('attachment/'.length)}`;
  }

  // 同じ report/ 内の別レポートへのリンクはサイト内のページへ
  const mdLink = clean.match(/^([^/#?]+)\.md(#.*)?$/);
  if (mdLink) return `${base}/reports/${mdLink[1]}/${mdLink[2] ?? ''}`;

  // それ以外の相対リンクは GitHub 上の report/ ディレクトリを指すものとして扱う
  return `${base}/report-assets/${slug}/${clean}`;
}

/** 表を横スクロールできるようにラップし、相対 URL を base 付きに書き換える。 */
function rewritePlugin(options: RewriteOptions) {
  return () => (tree: HastNode) => {
    const walk = (node: HastNode): HastNode => {
      if (node.children) node.children = node.children.map(walk);

      if (node.type !== 'element' || !node.properties) return node;

      if (node.tagName === 'a' && typeof node.properties.href === 'string') {
        node.properties.href = rewriteUrl(node.properties.href, options);
      }
      if (node.tagName === 'img' && typeof node.properties.src === 'string') {
        node.properties.src = rewriteUrl(node.properties.src, options);
        node.properties.loading = 'lazy';
        node.properties.decoding = 'async';
      }
      if (node.tagName === 'table') {
        return {
          type: 'element',
          tagName: 'div',
          properties: { className: ['table-scroll'] },
          children: [node],
        };
      }
      return node;
    };

    walk(tree);
  };
}

const schema = {
  ...defaultSchema,
  attributes: {
    ...defaultSchema.attributes,
    '*': [...(defaultSchema.attributes?.['*'] ?? []), 'align', 'style'],
  },
};

/**
 * Markdown を HTML にする。生 HTML は remark-rehype の既定で捨て、
 * さらに rehype-sanitize を通すので、レポート内の script などは実行されない。
 */
export async function renderMarkdown(markdown: string, options: RewriteOptions): Promise<string> {
  const file = await unified()
    .use(remarkParse)
    .use(remarkGfm)
    .use(remarkRehype)
    .use(rehypeSanitize, schema)
    .use(rewritePlugin(options))
    .use(rehypeStringify)
    .process(markdown);

  return String(file);
}
