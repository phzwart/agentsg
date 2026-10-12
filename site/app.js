const stage = document.getElementById("stage");
const canvas = document.getElementById("cloud");
const svg = d3.select("#axes");
const ctx = canvas.getContext("2d", { alpha: false });

let meta, xy, sys, cent, grid;
let encoder = "";
let colorBy = "system";
let hiddenSys = new Set();
let hiddenCent = new Set();
let focus = -1;
let selectMode = "one";
let lassoScreen = null;
let lassoWorld = null;
let lassoIdx = [];
let lassoIds = [];
let lassoUni = null;
let uniHidden = new Set();
let uniToken = 0;
let lassoColor = "uniprot";
let xBase, yBase, xScale, yScale;
let cellView = null;

function resize() {
  const r = stage.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = r.width * dpr;
  canvas.height = r.height * dpr;
  svg.attr("viewBox", `0 0 ${r.width} ${r.height}`).attr("width", r.width).attr("height", r.height);
  d3.select("#pick").attr("viewBox", `0 0 ${r.width} ${r.height}`).attr("width", r.width).attr("height", r.height);
  if (xy) fit(); else draw();
}

function dataExtent() {
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  for (let i = 0; i < meta.n; i++) {
    const x = xy[i * 2], y = xy[i * 2 + 1];
    if (x < xmin) xmin = x;
    if (x > xmax) xmax = x;
    if (y < ymin) ymin = y;
    if (y > ymax) ymax = y;
  }
  return { xmin, xmax, ymin, ymax };
}

function fit() {
  const { xmin, xmax, ymin, ymax } = dataExtent();
  const w = canvas.clientWidth, h = canvas.clientHeight;
  const dx = Math.max(xmax - xmin, 1e-6), dy = Math.max(ymax - ymin, 1e-6);
  xBase = d3.scaleLinear().domain([xmin - 0.04 * dx, xmax + 0.04 * dx]).range([48, w - 16]);
  yBase = d3.scaleLinear().domain([ymin - 0.04 * dy, ymax + 0.04 * dy]).range([h - 36, 16]);
  d3.select(stage).call(zoom.transform, d3.zoomIdentity);
}

function buildGrid() {
  const n = meta.n;
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = xy[i * 2], y = xy[i * 2 + 1];
    if (x < xmin) xmin = x;
    if (x > xmax) xmax = x;
    if (y < ymin) ymin = y;
    if (y > ymax) ymax = y;
  }
  const cols = 180;
  const cell = Math.max(xmax - xmin, ymax - ymin, 1e-6) / cols;
  const bins = new Map();
  for (let i = 0; i < n; i++) {
    const key = Math.floor(xy[i * 2] / cell) + "," + Math.floor(xy[i * 2 + 1] / cell);
    let bin = bins.get(key);
    if (!bin) bins.set(key, bin = []);
    bin.push(i);
  }
  grid = { bins, cell };
}

function nearest(px, py, pixel) {
  const wx = xScale.invert(px), wy = yScale.invert(py);
  const r = Math.abs(xScale.invert(px + pixel) - wx);
  const c = grid.cell;
  const i0 = Math.floor((wx - r) / c), i1 = Math.floor((wx + r) / c);
  const j0 = Math.floor((wy - r) / c), j1 = Math.floor((wy + r) / c);
  let best = -1, bestD = r * r;
  for (let i = i0; i <= i1; i++) {
    for (let j = j0; j <= j1; j++) {
      const bin = grid.bins.get(i + "," + j);
      if (!bin) continue;
      for (const p of bin) {
        if (hiddenSys.has(sys[p]) || hiddenCent.has(cent[p])) continue;
        const dx = xy[p * 2] - wx, dy = xy[p * 2 + 1] - wy;
        const d = dx * dx + dy * dy;
        if (d < bestD) { bestD = d; best = p; }
      }
    }
  }
  return best;
}

function palette() {
  return colorBy === "centring"
    ? { codes: cent, names: meta.centerings, colors: meta.centeringColors }
    : { codes: sys, names: meta.systems, colors: meta.systemColors };
}

function renderLegend() {
  const { names, colors } = palette();
  const hidden = colorBy === "centring" ? hiddenCent : hiddenSys;
  const box = document.getElementById("legend");
  box.replaceChildren();
  names.forEach((name, i) => {
    const span = document.createElement("span");
    if (hidden.has(i)) span.className = "off";
    const swatch = document.createElement("i");
    swatch.style.background = colors[i];
    span.append(swatch, document.createTextNode(name));
    box.appendChild(span);
  });
}

function draw() {
  const w = canvas.width, h = canvas.height;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, w, h);
  if (!xy) return;
  const { codes, colors } = palette();
  const dpr = window.devicePixelRatio || 1;
  const kx = xScale(1) - xScale(0);
  const ky = yScale(1) - yScale(0);
  ctx.setTransform(kx * dpr, 0, 0, ky * dpr, xScale(0) * dpr, yScale(0) * dpr);
  const kx0 = xBase(1) - xBase(0);
  const zoomK = kx0 ? Math.abs(kx / kx0) : 1;
  const dotPx = Math.min(9, 1.7 + 1.15 * Math.log2(Math.max(zoomK, 1))) / 4;
  const s = dotPx / Math.abs(kx);
  for (let g = 0; g < colors.length; g++) {
    if (colorBy === "system" && hiddenSys.has(g)) continue;
    if (colorBy === "centring" && hiddenCent.has(g)) continue;
    ctx.fillStyle = colors[g];
    ctx.beginPath();
    for (let i = 0; i < meta.n; i++) {
      if (codes[i] !== g) continue;
      if (hiddenSys.has(sys[i]) || hiddenCent.has(cent[i])) continue;
      ctx.rect(xy[i * 2], xy[i * 2 + 1], s, s);
    }
    ctx.fill();
  }
  if (lassoUni && lassoUni.codes.length === lassoIdx.length) {
    const s2 = s * 2.1;
    const order = lassoUni.groups.slice().reverse();
    for (const group of order) {
      if (uniHidden.has(group.code)) continue;
      ctx.fillStyle = group.color;
      ctx.beginPath();
      for (let k = 0; k < lassoIdx.length; k++) {
        if (lassoUni.codes[k] !== group.code) continue;
        const i = lassoIdx[k];
        ctx.rect(xy[i * 2] - s2 * 0.25, xy[i * 2 + 1] - s2 * 0.25, s2, s2);
      }
      ctx.fill();
    }
  } else if (lassoIdx.length && lassoIdx.length < 20000) {
    ctx.fillStyle = "#111";
    ctx.beginPath();
    const s2 = s * 1.8;
    for (const i of lassoIdx) {
      if (hiddenSys.has(sys[i]) || hiddenCent.has(cent[i])) continue;
      ctx.rect(xy[i * 2] - s2 * 0.25, xy[i * 2 + 1] - s2 * 0.25, s2, s2);
    }
    ctx.fill();
  }
  if (focus >= 0 && selectMode === "one" && !hiddenSys.has(sys[focus]) && !hiddenCent.has(cent[focus])) {
    ctx.strokeStyle = "#111";
    ctx.lineWidth = 1.6 / Math.abs(xScale(1) - xScale(0));
    const x = xy[focus * 2], y = xy[focus * 2 + 1];
    ctx.beginPath();
    ctx.moveTo(x - 4 * s, y - 4 * s); ctx.lineTo(x + 4 * s, y + 4 * s);
    ctx.moveTo(x - 4 * s, y + 4 * s); ctx.lineTo(x + 4 * s, y - 4 * s);
    ctx.stroke();
  }
  drawAxes();
  drawLasso();
}

function drawAxes() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  const sx = xScale, sy = yScale;
  svg.selectAll("*").remove();
  svg.append("g").attr("transform", `translate(0,${h - 28})`)
    .call(d3.axisBottom(sx).ticks(5).tickSizeOuter(0))
    .call(g => g.select(".domain").attr("stroke", "#ccc"))
    .call(g => g.selectAll("text").attr("fill", "#666"));
  svg.append("g").attr("transform", "translate(40,0)")
    .call(d3.axisLeft(sy).ticks(5).tickSizeOuter(0))
    .call(g => g.select(".domain").attr("stroke", "#ccc"))
    .call(g => g.selectAll("text").attr("fill", "#666"));
  svg.append("text").attr("x", w / 2).attr("y", h - 6).attr("text-anchor", "middle")
    .attr("fill", "#666").attr("font-size", 12).text("plane x");
  svg.append("text").attr("transform", "rotate(-90)").attr("x", -h / 2).attr("y", 12)
    .attr("text-anchor", "middle").attr("fill", "#666").attr("font-size", 12).text("plane y");
}

function shownText() {
  let n = 0;
  for (let i = 0; i < meta.n; i++) {
    if (!hiddenSys.has(sys[i]) && !hiddenCent.has(cent[i])) n++;
  }
  document.getElementById("shown").textContent =
    n.toLocaleString() + " of " + meta.n.toLocaleString() + " cells are drawn. A cell stays when its crystal system and its centring are both ticked.";
}

function applyMode() {
  const lasso = selectMode === "lasso";
  stage.classList.toggle("lasso", lasso);
  document.getElementById("one").hidden = lasso;
  document.getElementById("search").hidden = lasso;
  document.getElementById("many").hidden = !lasso;
  draw();
}

function fmtAngle(x) {
  for (const t of [60, 90, 120]) {
    if (Math.abs(x - t) < 1e-3) return t.toFixed(2);
  }
  return x.toFixed(2);
}

const vadd = (u, v) => [u[0] + v[0], u[1] + v[1], u[2] + v[2]];
const vsub = (u, v) => [u[0] - v[0], u[1] - v[1], u[2] - v[2]];
const vmul = (v, s) => [v[0] * s, v[1] * s, v[2] * s];
const vdot = (u, v) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
const vcross = (u, v) => [
  u[1] * v[2] - u[2] * v[1],
  u[2] * v[0] - u[0] * v[2],
  u[0] * v[1] - u[1] * v[0],
];
const vlen = (v) => Math.hypot(v[0], v[1], v[2]);
const vunit = (v) => {
  const n = vlen(v);
  return n < 1e-12 ? [1, 0, 0] : vmul(v, 1 / n);
};
function vrodrigues(v, axis, ang) {
  const c = Math.cos(ang), s = Math.sin(ang);
  return vadd(vadd(vmul(v, c), vmul(vcross(axis, v), s)), vmul(axis, vdot(axis, v) * (1 - c)));
}

function cellFrame(va, vb, vc) {
  const axes = [va, vb, vc];
  const ranked = [0, 1, 2].sort((i, j) => vlen(axes[j]) - vlen(axes[i]) || i - j);
  const u1 = vunit(axes[ranked[0]]);
  const u2 = vunit(axes[ranked[1]]);
  let normal = vunit(vcross(u1, u2));
  if (vdot(axes[ranked[2]], normal) < 0) normal = vmul(normal, -1);
  // Look into the corner where the two long edges meet the short one,
  // with the long edges foreshortened by the same amount.
  const view = vunit(vadd(vadd(u1, u2), normal));
  let right = vsub(u1, vmul(view, vdot(u1, view)));
  right = vlen(right) < 1e-8 ? vunit(vsub(u2, vmul(view, vdot(u2, view)))) : vunit(right);
  let up = vunit(vcross(view, right));
  if (vdot(u2, up) < 0) up = vmul(up, -1);
  return { right, up, view };
}

function drawWire(body) {
  const a = body.a, b = body.b, c = body.c;
  const rad = Math.PI / 180;
  const ca = Math.cos(body.alpha * rad), cb = Math.cos(body.beta * rad), cg = Math.cos(body.gamma * rad);
  const sg = Math.sin(body.gamma * rad);
  const V = a * b * c * Math.sqrt(Math.max(0, 1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg));
  const va = [a, 0, 0];
  const vb = [b * cg, b * sg, 0];
  const vc = [c * cb, c * (ca - cb * cg) / sg, V / (a * b * sg)];
  const corners = [
    [0, 0, 0], va, vadd(va, vb), vb, vc,
    vadd(vc, va), vadd(vadd(vc, va), vb), vadd(vc, vb),
  ];
  const center = vmul(vadd(vadd(va, vb), vc), 0.5);
  let radius = 0;
  for (const p of corners) radius = Math.max(radius, vlen(vsub(p, center)));
  const home = cellFrame(va, vb, vc);
  cellView = {
    corners, center, radius,
    right: home.right.slice(), up: home.up.slice(), view: home.view.slice(),
    home, zoom: 1,
  };
  paintCell();
}

function paintCell() {
  if (!cellView) return;
  const { corners, center, radius, right, up, view, zoom } = cellView;
  const fig = d3.select("#cellfig");
  fig.selectAll("*").remove();
  const w = 300, h = 260, margin = 28;
  fig.attr("viewBox", `0 0 ${w} ${h}`).attr("preserveAspectRatio", "xMidYMid meet");
  const k = Math.min(w - 2 * margin, h - 2 * margin) / (2 * radius) * zoom;
  const P = corners.map((p) => {
    const d = vsub(p, center);
    return [w / 2 + vdot(d, right) * k, h / 2 - vdot(d, up) * k, vdot(d, view)];
  });
  const faces = [[0, 1, 2, 3], [0, 3, 7, 4], [0, 4, 5, 1], [5, 6, 7, 4], [1, 2, 6, 5], [2, 3, 7, 6]];
  const front = [];
  for (const idx of faces) {
    const mid = vmul(idx.reduce((s, i) => vadd(s, corners[i]), [0, 0, 0]), 1 / idx.length);
    let normal = vcross(vsub(corners[idx[1]], corners[idx[0]]), vsub(corners[idx[2]], corners[idx[0]]));
    if (vdot(normal, vsub(mid, center)) < 0) normal = vmul(normal, -1);
    normal = vunit(normal);
    const facing = vdot(normal, view);
    if (facing <= 0.02) continue;
    front.push({ idx, facing, depth: vdot(vsub(mid, center), view) });
  }
  front.sort((p, q) => p.depth - q.depth);
  for (const face of front) {
    const g = Math.round(242 - 48 * face.facing);
    fig.append("path")
      .attr("d", face.idx.map((i, n) => `${n ? "L" : "M"}${P[i][0].toFixed(2)},${P[i][1].toFixed(2)}`).join("") + "Z")
      .attr("fill", `rgb(${g},${g},${g})`)
      .attr("stroke", "none");
  }
  const edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
  const line = d3.line().x(d => d[0]).y(d => d[1]);
  for (const [i, j] of edges) {
    fig.append("path").attr("d", line([P[i], P[j]]))
      .attr("fill", "none")
      .attr("stroke", "#111")
      .attr("stroke-linecap", "round")
      .attr("stroke-width", 1.75);
  }
  const labels = [[1, "a"], [3, "b"], [4, "c"]];
  for (const [i, text] of labels) {
    const ox = P[i][0] - w / 2, oy = P[i][1] - h / 2;
    const n = Math.hypot(ox, oy) || 1;
    fig.append("text")
      .attr("x", P[i][0] + 12 * ox / n).attr("y", P[i][1] + 12 * oy / n)
      .attr("text-anchor", "middle").attr("dominant-baseline", "central")
      .attr("fill", "#222").attr("font-size", 13).attr("font-family", "Helvetica, Arial, sans-serif")
      .text(text);
  }
  fig.append("circle")
    .attr("cx", P[0][0]).attr("cy", P[0][1]).attr("r", 3.5)
    .attr("fill", "#111");
}

function orbitCell(dx, dy) {
  if (!cellView) return;
  const k = 0.012;
  let { right, up, view } = cellView;
  right = vrodrigues(right, up, -dx * k);
  view = vrodrigues(view, up, -dx * k);
  up = vrodrigues(up, right, -dy * k);
  view = vrodrigues(view, right, -dy * k);
  view = vunit(view);
  right = vunit(vcross(up, view));
  up = vunit(vcross(view, right));
  cellView.right = right;
  cellView.up = up;
  cellView.view = view;
}

function bindCellFig() {
  const fig = document.getElementById("cellfig");
  let drag = null;
  fig.addEventListener("pointerdown", (ev) => {
    if (!cellView || ev.button !== 0) return;
    drag = { x: ev.clientX, y: ev.clientY };
    fig.setPointerCapture(ev.pointerId);
    fig.classList.add("dragging");
  });
  fig.addEventListener("pointermove", (ev) => {
    if (!drag) return;
    orbitCell(ev.clientX - drag.x, ev.clientY - drag.y);
    drag.x = ev.clientX;
    drag.y = ev.clientY;
    paintCell();
  });
  const endDrag = () => {
    drag = null;
    fig.classList.remove("dragging");
  };
  fig.addEventListener("pointerup", endDrag);
  fig.addEventListener("pointercancel", endDrag);
  fig.addEventListener("dblclick", () => {
    if (!cellView) return;
    cellView.right = cellView.home.right.slice();
    cellView.up = cellView.home.up.slice();
    cellView.view = cellView.home.view.slice();
    cellView.zoom = 1;
    paintCell();
  });
  fig.addEventListener("wheel", (ev) => {
    if (!cellView) return;
    ev.preventDefault();
    cellView.zoom *= ev.deltaY < 0 ? 1.08 : 1 / 1.08;
    cellView.zoom = Math.min(3, Math.max(0.45, cellView.zoom));
    paintCell();
  }, { passive: false });
}

function externalLink(href, text) {
  const link = document.createElement("a");
  link.href = href;
  link.target = "_blank";
  link.rel = "noopener";
  link.textContent = text;
  link.addEventListener("click", (ev) => ev.stopPropagation());
  return link;
}

function uniprotLink(acc) {
  return externalLink(
    "https://www.uniprot.org/uniprotkb/" + encodeURIComponent(acc) + "/",
    acc,
  );
}

function depositionGroupLink(label) {
  const id = label.startsWith("DEPOSITION_GROUP_") ? label.slice("DEPOSITION_GROUP_".length) : label;
  return externalLink("https://www.rcsb.org/groups/summary/entry/" + encodeURIComponent(id), label);
}

function renderCell(body) {
  const card = document.getElementById("cellcard");
  card.hidden = false;
  const title = document.getElementById("celltitle");
  title.replaceChildren();
  const link = document.createElement("a");
  link.href = "https://www.rcsb.org/structure/" + encodeURIComponent(body.pdb_id);
  link.target = "_blank";
  link.rel = "noopener";
  link.textContent = body.pdb_id;
  title.appendChild(link);
  const setting = document.getElementById("cellsetting");
  setting.textContent = "Selling-reduced primitive cell of deposited " + body.deposited + ". Edges are divided by the cube root of that primitive volume, so this cell has volume 1.";
  const rows = [
    ["a", body.a.toFixed(4), "α", fmtAngle(body.alpha) + "°"],
    ["b", body.b.toFixed(4), "β", fmtAngle(body.beta) + "°"],
    ["c", body.c.toFixed(4), "γ", fmtAngle(body.gamma) + "°"],
  ];
  const tb = document.getElementById("cellparams");
  tb.replaceChildren();
  for (const [k1, v1, k2, v2] of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${k1}</td><td class="num">${v1}</td><td>${k2}</td><td class="num">${v2}</td>`;
    tb.appendChild(tr);
  }
  drawWire(body);
  const proteins = document.getElementById("celluni");
  const items = body.uniprots || [];
  proteins.replaceChildren();
  if (items.length) {
    proteins.append("UniProt, longest first: ");
    items.forEach((item, i) => {
      if (i) proteins.append("; ");
      proteins.append((item.name || item.acc) + " (");
      if (item.acc) proteins.append(uniprotLink(item.acc));
      proteins.append(", " + item.n_res + ")");
    });
  } else {
    proteins.append("No UniProt on this entry.");
  }
  if (body.group_label) {
    proteins.append(document.createElement("br"));
    proteins.append(depositionGroupLink(body.group_label));
  }
}

function inside(x, y, poly) {
  let c = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i][0], yi = poly[i][1];
    const xj = poly[j][0], yj = poly[j][1];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) c = !c;
  }
  return c;
}

function drawLasso() {
  const pick = d3.select("#pick");
  pick.selectAll("*").remove();
  let pts = null;
  let closed = false;
  if (lassoScreen && lassoScreen.length > 1) {
    pts = lassoScreen;
    closed = false;
  } else if (lassoWorld && lassoWorld.length > 2 && xScale) {
    pts = lassoWorld.map(p => [xScale(p[0]), yScale(p[1])]);
    closed = true;
  }
  if (!pts) return;
  const line = d3.line();
  const path = closed ? pts.concat([pts[0]]) : pts;
  pick.append("path").attr("d", line(path))
    .attr("fill", closed ? "rgba(17,17,17,0.06)" : "none")
    .attr("stroke", "#111").attr("stroke-width", 1.25);
}

function finishLasso() {
  if (!lassoScreen || lassoScreen.length < 3) {
    lassoScreen = null;
    drawLasso();
    return;
  }
  lassoWorld = lassoScreen.map(p => [xScale.invert(p[0]), yScale.invert(p[1])]);
  lassoScreen = null;
  lassoIdx = [];
  for (let i = 0; i < meta.n; i++) {
    if (hiddenSys.has(sys[i]) || hiddenCent.has(cent[i])) continue;
    if (inside(xy[i * 2], xy[i * 2 + 1], lassoWorld)) lassoIdx.push(i);
  }
  lassoIds = lassoIdx.map(i => meta.pdb_id[i]);
  uniHidden = new Set();
  document.getElementById("ids").value = "";
  document.getElementById("idcount").textContent =
    lassoIds.length.toLocaleString() + " PDB ids inside the loop. Tick labels to list them.";
  colorLasso();
}

function chosenLassoIds() {
  if (!lassoUni || lassoUni.codes.length !== lassoIds.length || uniHidden.size === 0) return lassoIds;
  const out = [];
  for (let k = 0; k < lassoIds.length; k++) {
    if (!uniHidden.has(lassoUni.codes[k])) out.push(lassoIds[k]);
  }
  return out;
}

function writeLassoIds() {
  const ids = chosenLassoIds();
  const n = ids.length;
  const total = lassoIds.length;
  const count = document.getElementById("idcount");
  const noun = n === 1 ? "PDB id" : "PDB ids";
  count.textContent = uniHidden.size
    ? n.toLocaleString() + " " + noun + " from the ticked labels, of " + total.toLocaleString() + " inside the loop."
    : total.toLocaleString() + " PDB ids inside the loop. Unticked systems and centrings were left out.";
  const ta = document.getElementById("ids");
  ta.value = n <= 20000 ? ids.join("\n") : ids.slice(0, 200).join("\n") + "\n… " + n.toLocaleString() + " ids; use Download for the full list";
}

function renderUniLegend(body) {
  const box = document.getElementById("unilegend");
  const note = document.getElementById("uninote");
  const actions = document.getElementById("uniactions");
  box.replaceChildren();
  if (!body || !body.groups.length) {
    actions.hidden = true;
    note.textContent = body ? "No cells in the loop." : "";
    return;
  }
  actions.hidden = false;
  note.textContent = "Tick the labels you want. Their PDB ids are listed below, and Copy uses that list. Several labels can be on at once.";
  for (const group of body.groups) {
    const row = document.createElement("label");
    const tick = document.createElement("input");
    tick.type = "checkbox";
    tick.checked = !uniHidden.has(group.code);
    if (!tick.checked) row.className = "off";
    const swatch = document.createElement("i");
    swatch.style.background = group.color;
    tick.addEventListener("change", () => {
      if (tick.checked) uniHidden.delete(group.code);
      else uniHidden.add(group.code);
      renderUniLegend(lassoUni);
      writeLassoIds();
      draw();
    });
    row.append(tick, swatch);
    if (group.acc && group.acc.startsWith("DEPOSITION_GROUP_")) {
      row.append(depositionGroupLink(group.acc));
    } else if (group.acc) {
      row.append(document.createTextNode((group.name || group.acc) + "  "), uniprotLink(group.acc));
    } else {
      row.append(document.createTextNode(group.name || ""));
    }
    row.append(document.createTextNode("  " + group.n.toLocaleString()));
    box.appendChild(row);
  }
}

const UNIPROT_COLORS = [
  "#4c78a8", "#f58518", "#e45756", "#54a24b", "#b279a2", "#72b7b2",
  "#eeca3b", "#ff9da6", "#9d755d", "#1f77b4", "#2ca02c", "#9467bd",
];
const UNIPROT_OTHER = -2;
const UNIPROT_NONE = -1;

function classifyLabels(labels, kind) {
  const otherName = kind === "uniprot" ? "other proteins" : "other groups";
  const noneName = kind === "uniprot" ? "no UniProt" : "no deposition group";
  const counts = new Map();
  const names = new Map();
  for (const [acc, name] of labels) {
    counts.set(acc, (counts.get(acc) || 0) + 1);
    if (acc && !names.has(acc)) names.set(acc, name || acc);
  }
  const ranked = [...counts.keys()].filter((acc) => acc).sort((a, b) => {
    const diff = counts.get(b) - counts.get(a);
    if (diff) return diff;
    return a < b ? -1 : a > b ? 1 : 0;
  });
  const colorOf = new Map(ranked.slice(0, 12).map((acc, code) => [acc, code]));
  const groups = [];
  for (const [acc, code] of colorOf) {
    groups.push({ acc, name: names.get(acc), n: counts.get(acc), code, color: UNIPROT_COLORS[code] });
  }
  const otherN = ranked.slice(12).reduce((sum, acc) => sum + counts.get(acc), 0);
  if (otherN) groups.push({ acc: "", name: otherName, n: otherN, code: UNIPROT_OTHER, color: "#d0d0d0" });
  const noneN = counts.get("") || 0;
  if (noneN) groups.push({ acc: "", name: noneName, n: noneN, code: UNIPROT_NONE, color: "#666666" });
  const codes = labels.map(([acc]) => {
    if (!acc) return UNIPROT_NONE;
    return colorOf.has(acc) ? colorOf.get(acc) : UNIPROT_OTHER;
  });
  return { groups, codes };
}

function colorLasso() {
  lassoUni = null;
  uniHidden = new Set();
  const note = document.getElementById("uninote");
  document.getElementById("uniactions").hidden = true;
  document.getElementById("unilegend").replaceChildren();
  if (!lassoIds.length) {
    note.textContent = "";
    draw();
    return;
  }
  const labels = lassoIdx.map((i) => lassoColor === "deposit"
    ? [meta.deposit[i], meta.deposit[i]]
    : [meta.uniAcc[i], meta.uniName[i] || meta.uniAcc[i]]);
  lassoUni = classifyLabels(labels, lassoColor);
  uniHidden = new Set(lassoUni.groups.map((group) => group.code));
  renderUniLegend(lassoUni);
  writeLassoIds();
  draw();
}

function useEncoder(name) {
  encoder = name;
  const off = meta.xyOffset[name];
  xy = new Float32Array(pointsBuf, off, meta.n * 2);
  buildGrid();
  fit();
}

let showToken = 0;
let idIndex = null;
let cellEdges, proteinOff, proteinAccId, proteinNameId, proteinNres, proteinAcc, proteinName;

function littleEndian() {
  return new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;
}

function i32(buf, byteOffset, count) {
  if (littleEndian()) return new Int32Array(buf, byteOffset, count);
  const view = new DataView(buf);
  const out = new Int32Array(count);
  for (let i = 0; i < count; i++) out[i] = view.getInt32(byteOffset + i * 4, true);
  return out;
}

function proteinsAt(index) {
  const off = proteinOff[index], end = proteinOff[index + 1];
  const items = [];
  for (let k = off; k < end; k++) {
    items.push({
      acc: proteinAcc[proteinAccId[k]],
      name: proteinName[proteinNameId[k]],
      n_res: proteinNres[k],
    });
  }
  return items;
}

function showId(id) {
  if (selectMode !== "one") {
    selectMode = "one";
    document.querySelector('input[name="select"][value="one"]').checked = true;
    applyMode();
  }
  const status = document.getElementById("status");
  const key = id.trim().toUpperCase();
  const index = idIndex.get(key);
  if (index === undefined) {
    status.textContent = (key || "empty id") + " is not in the archive";
    return;
  }
  status.textContent = "";
  focus = index;
  renderCell({
    pdb_id: meta.pdb_id[index],
    deposited: meta.sg_hm[index],
    a: cellEdges[index * 6],
    b: cellEdges[index * 6 + 1],
    c: cellEdges[index * 6 + 2],
    alpha: cellEdges[index * 6 + 3],
    beta: cellEdges[index * 6 + 4],
    gamma: cellEdges[index * 6 + 5],
    uniprots: proteinsAt(index),
    group_label: meta.deposit[index],
  });
  draw();
}

let pointsBuf;
let zoom;
async function sha256Hex(buf) {
  const dig = await crypto.subtle.digest("SHA-256", buf);
  return [...new Uint8Array(dig)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

async function loadChecked(manifest, role) {
  const spec = manifest.files[role][0];
  const res = await fetch("data/" + spec.name);
  if (!res.ok) throw new Error(spec.name + " failed to load");
  const buf = await res.arrayBuffer();
  const got = await sha256Hex(buf);
  if (got !== spec.sha256) throw new Error(spec.name + " hash does not match the manifest");
  return buf;
}

async function main() {
  const status = document.getElementById("status");
  try {
    const manifest = await (await fetch("data/manifest.json")).json();
    meta = JSON.parse(new TextDecoder().decode(await loadChecked(manifest, "meta")));
    pointsBuf = await loadChecked(manifest, "points");
    cellEdges = new Float32Array(await loadChecked(manifest, "cells"));
    const protBuf = await loadChecked(manifest, "proteins");
    proteinAcc = JSON.parse(new TextDecoder().decode(await loadChecked(manifest, "protein_acc")));
    proteinName = JSON.parse(new TextDecoder().decode(await loadChecked(manifest, "protein_name")));
    const n = meta.n;
    const m = meta.proteins.m;
    proteinOff = i32(protBuf, 0, n + 1);
    proteinAccId = i32(protBuf, (n + 1) * 4, m);
    proteinNameId = i32(protBuf, (n + 1) * 4 + m * 4, m);
    proteinNres = i32(protBuf, (n + 1) * 4 + m * 8, m);
  } catch (err) {
    status.textContent = String(err.message || err);
    return;
  }
  idIndex = new Map(meta.pdb_id.map((pid, i) => [String(pid).toUpperCase(), i]));
  document.getElementById("foot").textContent =
    "Built " + meta.built + ". PDB snapshot " + meta.snapshot + ". agentsg " + meta.agentsg + ".";
  const base = 4;
  sys = new Uint8Array(pointsBuf, base, meta.n);
  cent = new Uint8Array(pointsBuf, base + meta.n, meta.n);

  const enc = document.getElementById("encoders");
  meta.encoders.forEach((name, i) => {
    const label = document.createElement("label");
    label.innerHTML = `<input type="radio" name="encoder" value="${name}" ${i === 0 ? "checked" : ""}> ${name}`;
    enc.appendChild(label);
  });
  enc.addEventListener("change", (ev) => {
    if (ev.target.name === "encoder") useEncoder(ev.target.value);
  });

  const sysBox = document.getElementById("systems");
  meta.systems.forEach((name, i) => {
    const label = document.createElement("label");
    label.innerHTML = `<input type="checkbox" data-sys="${i}" checked> ${name}`;
    sysBox.appendChild(label);
  });
  sysBox.addEventListener("change", (ev) => {
    const i = +ev.target.dataset.sys;
    if (ev.target.checked) hiddenSys.delete(i); else hiddenSys.add(i);
    shownText();
    renderLegend();
    draw();
  });

  const centBox = document.getElementById("centrings");
  meta.centerings.forEach((name, i) => {
    const label = document.createElement("label");
    label.innerHTML = `<input type="checkbox" data-cent="${i}" checked> ${name}`;
    centBox.appendChild(label);
  });
  centBox.addEventListener("change", (ev) => {
    const i = +ev.target.dataset.cent;
    if (ev.target.checked) hiddenCent.delete(i); else hiddenCent.add(i);
    shownText();
    renderLegend();
    draw();
  });
  document.querySelectorAll('input[name="select"]').forEach((el) => {
    el.addEventListener("change", () => {
      selectMode = el.value;
      lassoScreen = null;
      applyMode();
    });
  });
  document.getElementById("uniall").addEventListener("click", () => {
    uniHidden = new Set();
    renderUniLegend(lassoUni);
    writeLassoIds();
    draw();
  });
  document.getElementById("uninone").addEventListener("click", () => {
    uniHidden = new Set((lassoUni ? lassoUni.groups : []).map((group) => group.code));
    renderUniLegend(lassoUni);
    writeLassoIds();
    draw();
  });
  document.getElementById("copyids").addEventListener("click", async () => {
    const ids = chosenLassoIds();
    const text = ids.join("\n");
    try {
      await navigator.clipboard.writeText(text);
      document.getElementById("idcount").textContent = ids.length.toLocaleString() + " PDB ids copied.";
    } catch (err) {
      document.getElementById("ids").focus();
      document.getElementById("ids").select();
    }
  });
  document.getElementById("downloadids").addEventListener("click", () => {
    const blob = new Blob([chosenLassoIds().join("\n") + "\n"], { type: "text/plain" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "pdb_ids.txt";
    a.click();
    URL.revokeObjectURL(a.href);
  });
  document.querySelectorAll('input[name="color"]').forEach((el) => {
    el.addEventListener("change", () => { colorBy = el.value; renderLegend(); draw(); });
  });
  document.querySelectorAll('input[name="lassocolor"]').forEach((el) => {
    el.addEventListener("change", () => {
      lassoColor = el.value;
      if (selectMode === "lasso" && lassoIds.length) colorLasso();
    });
  });
  document.getElementById("search").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const id = document.getElementById("pdb").value.trim();
    if (id) showId(id);
  });

  zoom = d3.zoom().scaleExtent([0.4, 4000]).filter((event) => {
    if (selectMode === "lasso" && event.type !== "wheel") return false;
    return (!event.ctrlKey || event.type === "wheel") && !event.button;
  }).on("zoom", (ev) => {
    xScale = ev.transform.rescaleX(xBase);
    yScale = ev.transform.rescaleY(yBase);
    draw();
  });
  d3.select(stage).call(zoom);
  stage.addEventListener("mousemove", (ev) => {
    if (!grid || selectMode !== "one") return;
    const rect = canvas.getBoundingClientRect();
    const i = nearest(ev.clientX - rect.left, ev.clientY - rect.top, 8);
    document.getElementById("hover").textContent = i < 0 ? "" : `${meta.pdb_id[i]}  ${meta.sg_hm[i]}`;
  });
  stage.addEventListener("click", (ev) => {
    if (!grid || selectMode !== "one") return;
    const rect = canvas.getBoundingClientRect();
    const i = nearest(ev.clientX - rect.left, ev.clientY - rect.top, 10);
    if (i >= 0) showId(meta.pdb_id[i]);
  });
  stage.addEventListener("pointerdown", (ev) => {
    if (selectMode !== "lasso" || ev.button !== 0 || !xScale) return;
    const rect = canvas.getBoundingClientRect();
    lassoScreen = [[ev.clientX - rect.left, ev.clientY - rect.top]];
    stage.setPointerCapture(ev.pointerId);
  });
  stage.addEventListener("pointermove", (ev) => {
    if (!lassoScreen) return;
    const rect = canvas.getBoundingClientRect();
    const p = [ev.clientX - rect.left, ev.clientY - rect.top];
    const prev = lassoScreen[lassoScreen.length - 1];
    if ((p[0] - prev[0]) ** 2 + (p[1] - prev[1]) ** 2 < 16) return;
    lassoScreen.push(p);
    drawLasso();
  });
  stage.addEventListener("pointerup", () => {
    if (!lassoScreen) return;
    finishLasso();
  });

  window.addEventListener("resize", resize);
  resize();
  useEncoder(meta.encoders[0]);
  shownText();
  renderLegend();
}

bindCellFig();
main();
