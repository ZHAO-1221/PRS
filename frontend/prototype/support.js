// Minimal local runtime for the .dc.html screens exported from the Design canvas.
// shortcut: supports only what these screens use ({{holes}}, sc-for, sc-if, on* events, DCLogic); swap for the canvas runtime if screens grow.
document.write(
  '<script src="https://unpkg.com/react@18.3.1/umd/react.production.min.js" crossorigin="anonymous"><\/script>' +
  '<script src="https://unpkg.com/react-dom@18.3.1/umd/react-dom.production.min.js" crossorigin="anonymous"><\/script>'
);

document.addEventListener('DOMContentLoaded', () => {
  const h = React.createElement;
  const host = document.querySelector('x-dc');
  const helmet = host.querySelector('helmet');
  if (helmet) { [...helmet.children].forEach((el) => document.head.appendChild(el)); helmet.remove(); }
  // build.mjs sets window.DC_SCREENS to bundle every screen into one page
  const bundled = window.DC_SCREENS;
  const screens = bundled || {
    [decodeURIComponent(location.pathname.split('/').pop())]: {
      html: host.innerHTML,
      code: document.querySelector('script[data-dc-script]').textContent
    }
  };
  const page = document.createElement('style');
  page.textContent = 'html{background:#e9e4d8}body{display:flex;justify-content:center;padding:24px 0}';
  document.head.appendChild(page);

  // HTML lowercases attribute names, so map the event attributes back to React's names
  const EVENTS = ['onClick', 'onChange', 'onPointerDown', 'onPointerMove', 'onPointerUp', 'onPointerCancel',
    'onPointerLeave', 'onLostPointerCapture', 'onWheel'];
  const eventName = Object.fromEntries(EVENTS.map((e) => [e.toLowerCase(), e]));
  const camel = (s) => s.replace(/-([a-z])/g, (_, c) => c.toUpperCase());

  const lookup = (path, scope) => {
    if (path === 'true') return true;
    if (path === 'false') return false;
    if (/^-?\d+(\.\d+)?$/.test(path)) return Number(path);
    return path.split('.').reduce((v, k) => (v == null ? v : v[k]), scope);
  };
  const interp = (str, scope) => {
    const whole = str.match(/^\s*\{\{\s*([\w.$]+)\s*\}\}\s*$/);
    if (whole) return lookup(whole[1], scope);
    return str.replace(/\{\{\s*([\w.$]+)\s*\}\}/g, (_, p) => { const v = lookup(p, scope); return v == null ? '' : String(v); });
  };
  const styleObj = (css) => {
    const out = {};
    css.split(';').forEach((decl) => {
      const i = decl.indexOf(':');
      if (i < 0) return;
      const prop = decl.slice(0, i).trim();
      out[prop.startsWith('--') ? prop : camel(prop)] = decl.slice(i + 1).trim();
    });
    return out;
  };

  const kids = (node, scope) => [...node.childNodes].map((c, i) => build(c, scope, i));
  function build(node, scope, key) {
    if (node.nodeType === 3) return interp(node.textContent, scope);
    if (node.nodeType !== 1) return null;
    const tag = node.localName;
    if (tag === 'sc-for') {
      const list = lookup(node.getAttribute('list').replace(/[{}\s]/g, ''), scope) || [];
      const as = node.getAttribute('as');
      return h(React.Fragment, { key }, list.map((item, i) =>
        h(React.Fragment, { key: i }, kids(node, { ...scope, [as]: item, $index: i }))));
    }
    if (tag === 'sc-if') {
      return interp(node.getAttribute('value'), scope) ? h(React.Fragment, { key }, kids(node, scope)) : null;
    }
    const props = { key };
    for (const { name, value } of node.attributes) {
      const v = interp(value, scope);
      if (eventName[name]) props[eventName[name]] = v;
      else if (name === 'style') props.style = styleObj(String(v));
      else if (name === 'class') props.className = v;
      else if (name === 'for') props.htmlFor = v;
      else if (tag === 'svg' && !name.startsWith('aria-')) props[camel(name)] = v;
      else props[name] = v;
    }
    // icons and the skeleton overlay are static SVG: render their insides as-is
    if (tag === 'svg') return h('svg', { ...props, dangerouslySetInnerHTML: { __html: node.innerHTML } });
    return h(tag, props, ...(node.childNodes.length ? kids(node, scope) : []));
  }

  const components = {};
  for (const [name, { html, code }] of Object.entries(screens)) {
    const tpl = document.createElement('template');
    tpl.innerHTML = html;
    const template = tpl.content;
    class DCLogic extends React.Component {
      render() { return h(React.Fragment, null, kids(template, this.renderVals())); }
    }
    components[name] = new Function('DCLogic', code + '\nreturn Component;')(DCLogic);
  }
  const names = Object.keys(components);
  const current = () => {
    const name = decodeURIComponent(location.hash.slice(1));
    return components[name] ? name : names.includes('Main.dc.html') ? 'Main.dc.html' : names[0];
  };

  host.replaceWith(Object.assign(document.createElement('div'), { id: 'root' }));
  const root = ReactDOM.createRoot(document.getElementById('root'));
  const show = () => { const name = current(); root.render(h(components[name], { key: name })); };
  if (bundled) {
    // links between screens switch screens inside this one page
    document.addEventListener('click', (e) => {
      const a = e.target.closest('a[href$=".dc.html"]');
      if (!a) return;
      e.preventDefault();
      location.hash = a.getAttribute('href');
      window.scrollTo(0, 0);
    });
    window.addEventListener('hashchange', show);
  }
  show();
});
