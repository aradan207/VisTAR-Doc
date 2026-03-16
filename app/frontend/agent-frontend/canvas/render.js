import { el } from '../ui/dom.js';
import { escHTML, pretty, safeToolText } from '../core/utils.js';
import { NODE_H } from '../core/constants.js';
import { buildLayout, withRootRecursive, edgePath } from './layout.js';
import { drawMinimap } from './minimap.js';

export function render(state) {
  if (!state?.nodes || Object.keys(state.nodes).length === 0) {
    el.status.textContent = 'No nodes to display.';
    el.nodesLayer.innerHTML = '';
    el.edgesSvg.innerHTML = '';
    el.finalContent.textContent = '';
    drawMinimap([], { width: 1000, height: 600 });
    return;
  }

  const label = (el.query?.value || 'User Request').trim();
  const view = withRootRecursive(state, label);

  // ── Pass 1: create DOM nodes, measure natural heights ──────────
  el.nodesLayer.innerHTML = '';
  const divMap = {};

  for (const [id, n] of Object.entries(view.nodes)) {
    const div = document.createElement('div');
    div.className = `node ${n.status}${id === view.__root__ ? ' root' : ''}`;
    // place off-screen at 0,0 so it can be measured
    div.style.left = '0px';
    div.style.top = '0px';
    div.style.visibility = 'hidden';

    const badge = `${n.children?.length || 0}·${n.tool_calls?.length || 0}`;
    const toolsHTML = (n.tool_calls || [])
      .map(
        (t) => `
      <div class="tool">
        <div class="tool-name">🧩 ${escHTML(t.tool_name)}</div>
        <div class="tool-result">${safeToolText(
          t.result ?? '(no result)',
          220
        )}</div>
      </div>`
      )
      .join('');

    div.innerHTML = `
      <div class="row">
        <span class="dot"></span>
        <div class="title" title="${escHTML(n.name || 'Untitled')}">${escHTML(
      n.name || 'Untitled'
    )}</div>
        <span class="badge" title="children·tools">${badge}</span>
      </div>
      <div class="desc">${escHTML(n.description || '(no result)')}</div>
      ${toolsHTML ? `<div class="tools">${toolsHTML}</div>` : ''}
    `;

    el.nodesLayer.appendChild(div);
    divMap[id] = { div, node: n };
  }

  // Force layout so offsetHeight is accurate
  const nodeHeights = {};
  for (const [id, { div }] of Object.entries(divMap)) {
    nodeHeights[id] = Math.max(div.offsetHeight, NODE_H);
  }

  // ── Pass 2: compute layout with measured heights, position nodes ──
  const { positions, width, height } = buildLayout(view, nodeHeights);

  // Position each node div
  for (const [id, { div }] of Object.entries(divMap)) {
    const pos = positions[id] || { x: 0, y: 0 };
    div.style.left = `${pos.x}px`;
    div.style.top = `${pos.y}px`;
    div.style.visibility = '';
  }

  // ── Edges ──────────────────────────────────────────────────────
  el.edgesSvg.innerHTML = '';
  el.edgesSvg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  el.edgesSvg.setAttribute('width', width);
  el.edgesSvg.setAttribute('height', height);

  const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
  const marker = document.createElementNS(
    'http://www.w3.org/2000/svg',
    'marker'
  );
  marker.setAttribute('id', 'arrow');
  marker.setAttribute('orient', 'auto');
  marker.setAttribute('markerWidth', '10');
  marker.setAttribute('markerHeight', '10');
  marker.setAttribute('refX', '8');
  marker.setAttribute('refY', '3');
  const ap = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  ap.setAttribute('d', 'M0,0 L8,3 L0,6 z');
  ap.setAttribute('fill', '#64748b');
  marker.appendChild(ap);
  defs.appendChild(marker);
  el.edgesSvg.appendChild(defs);

  const edgeIndex = new Map();
  for (const node of Object.values(view.nodes)) {
    for (const dep of node.depends_on || []) {
      const from = positions[dep],
        to = positions[node.id];
      if (!from || !to) continue;
      const fromH = nodeHeights[dep] || NODE_H;
      const toH = nodeHeights[node.id] || NODE_H;
      const p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      p.setAttribute('d', edgePath(from, to, fromH, toH));
      p.setAttribute('marker-end', 'url(#arrow)');
      p.setAttribute('fill', 'none');
      p.setAttribute('stroke', '#94a3b8');
      p.setAttribute('stroke-width', '1.6');
      p.setAttribute('stroke-dasharray', '5 5');
      p.setAttribute('opacity', '.9');
      el.edgesSvg.appendChild(p);
      edgeIndex.set(`${node.id}<-${dep}`, p);
    }
  }

  // ── Event listeners ────────────────────────────────────────────
  for (const [id, { div, node: n }] of Object.entries(divMap)) {
    div.addEventListener('mouseenter', () => {
      for (const d of n.depends_on || [])
        edgeIndex.get(`${id}<-${d}`)?.classList.add('active');
    });
    div.addEventListener('mouseleave', () => {
      for (const d of n.depends_on || [])
        edgeIndex.get(`${id}<-${d}`)?.classList.remove('active');
    });
    div.addEventListener('click', () => {
      import('../ui/inspector.js').then((m) => m.openInspectorRich(n));
    });
  }

  // final
  const nodeCount = Object.keys(view.nodes || {}).length;
  const edgeCount = Object.values(view.nodes || {}).reduce(
    (acc, node) => acc + (Array.isArray(node.depends_on) ? node.depends_on.length : 0),
    0
  );
  const summaryText = `Showing ${nodeCount} nodes and ${edgeCount} connections.`;

  if (state.final) {
    const raw =
      typeof state.final === 'string'
        ? state.final
        : state.final.answer ?? pretty(state.final);

    // Configure marked for proper image rendering
    if (window.marked) {
      window.marked.setOptions({
        breaks: true,
        gfm: true
      });
    }
    
    const html = window.marked ? window.marked.parse(raw) : raw;
    el.finalContent.innerHTML = html;
    
    // Add click handlers for images to open in new tab
    const images = el.finalContent.querySelectorAll('img');
    images.forEach(img => {
      img.style.cursor = 'pointer';
      img.addEventListener('click', () => {
        window.open(img.src, '_blank');
      });
      // Auto-repair broken image URLs: try adding .png if extension is missing
      img.addEventListener('error', () => {
        const src = img.src;
        const hasExt = /\.(png|jpe?g|gif|webp|svg|bmp)$/i.test(src);
        if (!hasExt && !img.dataset.retried) {
          img.dataset.retried = '1';
          img.src = src + '.png';
          return; // give the .png URL a chance to load
        }
        img.alt = `Failed to load: ${img.src}`;
        img.style.border = '2px dashed #dc2626';
        img.style.padding = '20px';
        img.style.background = '#fef2f2';
        console.error('Failed to load image:', img.src);
      });
    });

    el.status.textContent = `${summaryText} Final answer ready.`;
  } else {
    el.finalContent.textContent = 'No answer yet.';
    el.status.textContent = `${summaryText} Awaiting final answer.`;
  }

  if (state.benchmarkScores) {
    if (state.benchmarkScores.error) {
      el.benchmarkScores.innerHTML = `<span style="color:#ef4444;font-size:13px;padding:4px 8px;font-weight:500;">Benchmark Error: ${state.benchmarkScores.error}</span>`;
    } else {
      const scores = state.benchmarkScores;
      el.benchmarkScores.innerHTML = `
        <div class="benchmark-score-chip"><span class="benchmark-score-label">Context Recall:</span> ${scores.context_recall}</div>
        <div class="benchmark-score-chip"><span class="benchmark-score-label">Faithfulness:</span> ${scores.faithfulness}</div>
        <div class="benchmark-score-chip"><span class="benchmark-score-label">Fact. Correctness:</span> ${scores.factual_correctness}</div>
        <div class="benchmark-score-chip"><span class="benchmark-score-label">Answer Relevancy:</span> ${scores.answer_relevancy}</div>
      `;
    }
    el.benchmarkScores.classList.remove('hidden');
  } else if (state.benchmarkStatus) {
    el.benchmarkScores.innerHTML = `<span style="color:var(--text-muted);font-size:13px;padding:4px 8px;">${state.benchmarkStatus}</span>`;
    el.benchmarkScores.classList.remove('hidden');
  } else {
    el.benchmarkScores.innerHTML = '';
    el.benchmarkScores.classList.add('hidden');
  }

  drawMinimap(Object.values(positions), { width, height });
}
