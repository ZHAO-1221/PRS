// Bundles every *.dc.html screen, support.js and the images they use into ../prototype.html.
// Run: node frontend/prototype/build.mjs
import { readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { dirname, extname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const dir = dirname(fileURLToPath(import.meta.url));
const out = join(dir, '..', 'prototype.html');
const mime = { '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp' };

const dataUri = (rel) => {
  const file = resolve(dir, rel);
  const type = mime[extname(file).toLowerCase()];
  if (!type) throw new Error('Unsupported image type: ' + rel);
  return `data:${type};base64,${readFileSync(file).toString('base64')}`;
};
const inlineImages = (text) => text.replace(/\.\.\/assets\/[\w.-]+/g, dataUri);
const pick = (src, re, file) => {
  const m = src.match(re);
  if (!m) throw new Error(`${file}: missing ${re}`);
  return m[1];
};

let helmet = '';
const screens = {};
for (const file of readdirSync(dir).filter((f) => f.endsWith('.dc.html')).sort()) {
  const src = readFileSync(join(dir, file), 'utf8');
  const body = pick(src, /<x-dc>([\s\S]*)<\/x-dc>/, file);
  helmet ||= pick(body, /<helmet>([\s\S]*?)<\/helmet>/, file);
  screens[file] = {
    html: inlineImages(body.replace(/<helmet>[\s\S]*?<\/helmet>/, '')),
    code: inlineImages(pick(src, /<script type="text\/x-dc"[^>]*>([\s\S]*?)<\/script>/, file))
  };
}
if (!screens['Main.dc.html']) throw new Error('Main.dc.html not found');

// keep "</script>" inside the JSON from closing the inline script
const json = JSON.stringify(screens).replace(/</g, '\\u003c');
const runtime = readFileSync(join(dir, 'support.js'), 'utf8');
writeFileSync(out, `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dog behaviour app prototype</title>
${helmet.trim()}
<script>window.DC_SCREENS = ${json};</script>
<script>
${runtime}
</script>
</head>
<body>
<x-dc></x-dc>
</body>
</html>
`);
console.log(`Wrote ${out} (${Object.keys(screens).length} screens, ${(Buffer.byteLength(readFileSync(out)) / 1024).toFixed(0)} KB)`);
