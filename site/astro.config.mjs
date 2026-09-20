// @ts-check
import { defineConfig } from 'astro/config';

// GitHub Pages project site: https://jimoto-no-llm.github.io/bench-of-us/
// base を変えると生成される URL がすべて変わるので、フォークで公開する場合のみ調整する。
export default defineConfig({
  site: 'https://jimoto-no-llm.github.io',
  base: '/bench-of-us',
  trailingSlash: 'always',
  build: { format: 'directory' },
});
