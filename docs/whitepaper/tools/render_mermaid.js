// Render every mermaid/*.mmd to a tightly cropped vector PDF in tex/fig/.
// Usage: node tools/render_mermaid.js <mermaid.min.js>   (run from docs/whitepaper)
const { chromium } = require('playwright');
const fs = require('fs'), path = require('path');
(async () => {
  const lib = process.argv[2];
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 2000, height: 1200 } });
  for (const f of fs.readdirSync('mermaid').filter(f => f.endsWith('.mmd')).sort()) {
    const src = fs.readFileSync(path.join('mermaid', f), 'utf8');
    await page.setContent('<html><body style="margin:0;background:#fff"><div id="o" style="display:inline-block;padding:6px"></div></body></html>');
    await page.addScriptTag({ path: lib });
    await page.evaluate(async (s) => {
      mermaid.initialize({ startOnLoad: false, theme: 'neutral',
        themeVariables: { fontFamily: 'Helvetica, Arial, sans-serif', fontSize: '15px' },
        flowchart: { useMaxWidth: false }, sequence: { useMaxWidth: false }, state: { useMaxWidth: false } });
      const { svg } = await mermaid.render('g', s);
      document.getElementById('o').innerHTML = svg;
    }, src);
    const box = await page.locator('#o').boundingBox();
    const out = path.join('tex/fig', f.replace('.mmd', '.pdf'));
    await page.pdf({ path: out, width: Math.ceil(box.width) + 'px', height: Math.ceil(box.height) + 1 + 'px',
                     printBackground: true, pageRanges: '1' });
    console.log(out, Math.round(box.width), 'x', Math.round(box.height));
  }
  await browser.close();
})();
