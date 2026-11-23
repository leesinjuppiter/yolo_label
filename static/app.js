'use strict';

const byId = id => document.getElementById(id);
const canvas = byId('canvas');
const context = canvas.getContext('2d');
const canvasWrap = byId('canvas-wrap');
const zoom = byId('zoom');
const colors = {foreign: '#87ff9b', material: '#f0e493', selected: '#ffffff', half: '#5cddff'};
const neighborhoodFactor = 0.5;
const classes = {foreign: '异物（未细分）', rockbolt: '锚杆', mesh: '网片',
  board: '板状物', bottle: '瓶子', coal: '煤粒', other_material: '其他物料'};
const materialClasses = new Set(['coal', 'other_material']);
let batch, frame, image, selected, gesture, drawing;
let frames = [], busy = false, dirty = false, mode = 'select';
let editRevision = 0, pendingSave;
const hiddenBoxes = new Set();

function make(tag, text) {
  const el = document.createElement(tag);
  if (text !== undefined) el.textContent = text;
  return el;
}

function addOptions(select, entries) {
  select.replaceChildren(...entries.map(([value, text]) => {
    const option = make('option', text);
    option.value = value;
    return option;
  }));
}

function message(text, error = false) {
  byId('message').textContent = text;
  byId('message').dataset.error = String(error);
}

async function api(path, options) {
  const result = await fetch(path, options);
  const body = await result.json();
  if (!result.ok) throw new Error(body.error || body.message || '请求失败（' + result.status + '）');
  return body;
}

function framePath(id = frame.id) {
  return '/api/frames/' + encodeURIComponent(id);
}

async function run(action) {
  if (busy || gesture) return;
  busy = true;
  renderControls();
  message('');
  try {
    await action();
  } catch (error) {
    message(error.message, true);
  } finally {
    busy = false;
    renderControls();
  }
}

function renderControls() {
  const hasForeign = frame?.boxes.some(box => box.kind === 'foreign');
  byId('batch').disabled = busy || !batch;
  byId('review-controls').disabled = busy || !frame;
  byId('negative').disabled = hasForeign;
  byId('foreign-reviewed').disabled = Boolean(frame?.negative) || !hasForeign;
  byId('material-reviewed').disabled = !frame?.foreign_reviewed || !hasForeign;
  byId('prev').disabled = busy || !neighbor(-1);
  byId('next').disabled = busy || !neighbor(1);
  byId('export').disabled = busy || !batch;
}

function markDirty() {
  editRevision += 1;
  dirty = true;
  byId('save-state').textContent = '有未保存修改';
}

function syncStatus() {
  byId('foreign-reviewed').checked = frame.foreign_reviewed;
  byId('material-reviewed').checked = frame.material_reviewed;
  byId('negative').checked = frame.negative;
  renderControls();
}

function reviewedText(item) {
  if (item.negative) return '已确认无异物';
  if (item.material_reviewed) return '邻域已复核';
  if (item.foreign_reviewed) return '异物已复核';
  return '候选' + (item.candidate === 'positive' ? '阳性' : '阴性');
}

function matchesReviewFilter(item, filter) {
  if (filter === 'all') return true;
  if (filter === 'pending') return !item.foreign_reviewed;
  if (filter === 'material') return item.foreign_reviewed && !item.negative && !item.material_reviewed;
  if (filter === 'confirmed-negative') return item.negative;
  return item.candidate === filter;
}

function shownFrames() {
  const filter = byId('filter').value;
  const query = byId('frame-search').value.trim().toLowerCase();
  return frames.filter(item => matchesReviewFilter(item, filter)
    && (!query || item.name.toLowerCase().includes(query) || item.id.toLowerCase().includes(query)));
}

function neighbor(offset) {
  if (!frame) return;
  const list = shownFrames();
  const index = list.findIndex(item => item.id === frame.id);
  if (index >= 0) return list[index + offset];
  const current = frames.findIndex(item => item.id === frame.id);
  return offset > 0 ? list.find(item => frames.indexOf(item) > current)
    : list.findLast(item => frames.indexOf(item) < current);
}

function renderFrames() {
  const list = shownFrames().map(item => {
    const li = make('li');
    const button = make('button');
    button.type = 'button';
    button.setAttribute('aria-current', String(item.id === frame?.id));
    button.append(make('span', item.name), make('small', reviewedText(item)));
    button.addEventListener('click', () => navigate(item.id));
    li.append(button);
    return li;
  });
  if (!list.length) list.push(make('li', '当前筛选无图片'));
  byId('frame-list').replaceChildren(...list);
  byId('progress').textContent = '异物复核 ' + frames.filter(item => item.foreign_reviewed).length + ' / ' + frames.length;
}

async function showFrame(id) {
  const record = await api(framePath(id));
  const photo = new Image();
  photo.src = framePath(id) + '/image';
  await photo.decode();
  frame = record;
  image = photo;
  selected = undefined;
  hiddenBoxes.clear();
  dirty = false;
  byId('save-state').textContent = frame.version ? '已保存' : '待复核';
  byId('frame-name').textContent = frame.name;
  byId('frame-meta').textContent = [frame.id, frame.split, reviewedText(frame)].join(' · ');
  byId('note').value = frame.note;
  syncStatus();
  renderFrames();
  canvasWrap.scrollTo(0, 0);
  sizeCanvas();
  updateRegions();
  history.replaceState(null, '', '?' + new URLSearchParams({batch, image: id}).toString());
}

async function openBatch(id, preferred) {
  if (dirty) await saveFrame();
  const result = await api('/api/frames?batch=' + encodeURIComponent(id));
  batch = id;
  frames = result.frames;
  addOptions(byId('batch'), result.batches.map(item => [item.id, item.name]));
  byId('batch').value = id;
  byId('frame-search').value = '';
  byId('filter').value = 'all';
  const item = frames.find(candidate => candidate.id === preferred)
    || frames.find(candidate => !candidate.foreign_reviewed) || frames[0];
  frame = image = selected = undefined;
  renderFrames();
  draw();
  if (item) await showFrame(item.id);
}

function navigate(id) {
  if (frame?.id === id) return;
  run(async () => {
    if (dirty) await saveFrame();
    await showFrame(id);
  });
}

async function saveFrame(automatic = false) {
  if (pendingSave) await pendingSave;
  if (!frame || !dirty) return;
  const revision = editRevision;
  const {version, boxes, foreign_reviewed, material_reviewed, negative, verdict, note} = frame;
  pendingSave = api(framePath(), {method: 'PUT', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({version, boxes, foreign_reviewed, material_reviewed, negative, verdict, note})});
  try {
    const saved = await pendingSave;
    // Keep edits made while an automatic save was in flight.
    frame.version = saved.version;
    if (editRevision === revision) {
      dirty = false;
      byId('save-state').textContent = automatic ? '已自动保存' : '已保存';
    }
    frames[frames.findIndex(item => item.id === frame.id)] = saved;
    byId('frame-meta').textContent = [saved.id, saved.split, reviewedText(saved)].join(' · ');
    renderFrames();
  } finally {
    pendingSave = undefined;
  }
}

function editedKinds(...kinds) {
  if (kinds.includes('foreign')) {
    frame.foreign_reviewed = frame.material_reviewed = frame.negative = false;
    frame.verdict = '';
  } else {
    frame.material_reviewed = false;
  }
  syncStatus();
  markDirty();
}

function selectedBox() {
  return frame?.boxes.find(box => box.id === selected);
}

function renderLabels(focusToggleId) {
  if (!frame) return;
  const listed = frame.boxes.filter(matchesLayer);
  const visibleCount = listed.filter(isVisible).length;
  const toggleAll = byId('toggle-all-boxes');
  toggleAll.disabled = listed.length === 0;
  toggleAll.textContent = listed.length && !visibleCount ? '显示所有' : '隐藏所有';
  const box = selectedBox();
  byId('box-editor').hidden = !box || !isVisible(box);
  if (box && isVisible(box)) {
    byId('box-class').value = classes[box.subtype] ? box.subtype : box.kind === 'foreign' ? 'foreign' : 'other_material';
    byId('box-angle').value = (geometry(box).angle * 180 / Math.PI).toFixed(1);
  }
  byId('box-count').textContent = visibleCount + ' / ' + frame.boxes.length;
  let focusToggle;
  byId('box-list').replaceChildren(...listed.map(item => {
    const hidden = hiddenBoxes.has(item.id);
    const label = classes[item.subtype] || item.subtype;
    const row = make('div');
    row.className = 'box-row';
    const select = make('button', label);
    select.type = 'button';
    select.disabled = hidden;
    select.setAttribute('aria-pressed', String(selected === item.id));
    select.addEventListener('click', () => setSelection(item.id));
    const toggle = make('button', hidden ? '显示' : '隐藏');
    toggle.type = 'button';
    toggle.setAttribute('aria-label', (hidden ? '显示' : '隐藏') + label + '框');
    toggle.addEventListener('click', () => {
      if (hidden) hiddenBoxes.delete(item.id);
      else {
        hiddenBoxes.add(item.id);
        if (selected === item.id) selected = undefined;
      }
      renderLabels(item.id);
      draw();
    });
    if (focusToggleId === item.id) focusToggle = toggle;
    row.append(select, toggle);
    return row;
  }));
  focusToggle?.focus({preventScroll: true});
}

byId('toggle-all-boxes').addEventListener('click', () => {
  if (busy || gesture || !frame) return;
  const listed = frame.boxes.filter(matchesLayer);
  const hide = listed.some(isVisible);
  for (const box of listed) {
    if (hide) hiddenBoxes.add(box.id);
    else hiddenBoxes.delete(box.id);
  }
  if (hiddenBoxes.has(selected)) selected = undefined;
  updateCursor();
  renderLabels();
  draw();
});

function setSelection(id) {
  selected = id;
  renderLabels();
  draw();
}

function setMode(next) {
  mode = next;
  updateCursor();
  byId('select-tool').setAttribute('aria-pressed', String(mode === 'select'));
  byId('draw-tool').setAttribute('aria-pressed', String(mode === 'draw'));
}

function boxGap(a, b) {
  const aEdges = a.map((point, i) => [point, a[(i+1)%a.length]]);
  const bEdges = b.map((point, i) => [point, b[(i+1)%b.length]]);
  const separated = [...aEdges, ...bEdges].some(([[sx, sy], [ex, ey]]) => {
    const nx = sy - ey, ny = ex - sx;
    const aProjection = a.map(([x, y]) => x*nx + y*ny);
    const bProjection = b.map(([x, y]) => x*nx + y*ny);
    return Math.max(...aProjection) < Math.min(...bProjection)
      || Math.max(...bProjection) < Math.min(...aProjection);
  });
  if (!separated) return 0;

  let nearest = Infinity;
  for (const [points, edges] of [[a, bEdges], [b, aEdges]]) {
    for (const [x, y] of points) {
      for (const [[sx, sy], [ex, ey]] of edges) {
        const dx = ex - sx, dy = ey - sy;
        const fraction = ((x-sx)*dx + (y-sy)*dy) / (dx*dx + dy*dy);
        const position = Math.min(1, Math.max(0, fraction));
        nearest = Math.min(nearest, (x-sx-position*dx)**2 + (y-sy-position*dy)**2);
      }
    }
  }
  return Math.sqrt(nearest);
}

function neighborhood() {
  const foreigners = frame.boxes.filter(box => box.kind === 'foreign');
  const areas = foreigners.map(box => ({...geometry(box), points: pixels(box.points)}));
  const included = new Set();
  for (const box of frame.boxes) {
    if (box.kind !== 'material') continue;
    const points = pixels(box.points);
    if (areas.some(area => boxGap(points, area.points) <= neighborhoodFactor * Math.max(area.width, area.height) + 1e-9)) {
      included.add(box.id);
    }
  }
  return {areas, included};
}

let regions;
function updateRegions(refreshLabels = true) {
  if (!frame) return;
  regions = neighborhood();
  byId('count-half').textContent = regions.included.size;
  if (refreshLabels) renderLabels();
  draw();
}

function isVisible(box) {
  return !hiddenBoxes.has(box.id) && matchesLayer(box);
}

function matchesLayer(box) {
  if (box.kind === 'foreign') return true;
  const mode = byId('materials').value;
  return mode === 'all' || (mode === 'inner' && Boolean(regions?.included.has(box.id)));
}

function drawNeighborhood() {
  if (!regions || byId('region').value === 'off') return;
  for (const {center, width, height, angle} of regions.areas) {
    const radius = neighborhoodFactor * Math.max(width, height);
    context.save();
    context.translate(...center);
    context.rotate(angle);
    context.strokeStyle = colors.half;
    context.lineWidth = 1.6 * visualPixels();
    context.beginPath();
    context.roundRect(-width/2 - radius, -height/2 - radius, width + 2*radius, height + 2*radius, radius);
    context.stroke();
    context.restore();
  }
}

function draw() {
  context.clearRect(0, 0, canvas.width, canvas.height);
  if (!frame || !image) return;
  context.drawImage(image, 0, 0, canvas.width, canvas.height);
  drawNeighborhood();
  for (const box of frame.boxes) {
    if (box.id !== selected && isVisible(box)) {
      strokeBox(pixels(box.points), colors[box.kind], box.kind === 'foreign' ? classes[box.subtype] : null);
    }
  }
  const box = selectedBox();
  if (box && isVisible(box)) {
    const points = pixels(box.points);
    strokeBox(points, colors.selected, classes[box.subtype], true);
    drawHandles(points);
  }
  if (drawing) strokeBox(drawing, colors.selected, null, true, true);
}

function sizeCanvas() {
  if (!frame) return;
  if (canvas.width !== frame.width) canvas.width = frame.width;
  if (canvas.height !== frame.height) canvas.height = frame.height;
  const bounds = canvasWrap.getBoundingClientRect();
  const fit = Math.min((bounds.width - 2 * canvasWrap.clientLeft) / frame.width,
    (bounds.height - 2 * canvasWrap.clientTop) / frame.height);
  const value = Number(zoom.value);
  const scale = Math.max(.01, fit * value);
  canvas.style.width = (frame.width * scale) + 'px';
  canvas.style.height = (frame.height * scale) + 'px';
  byId('zoom-value').textContent = Math.round(value * 100) + '%';
  byId('zoom-out').disabled = value <= Number(zoom.min);
  byId('zoom-in').disabled = value >= Number(zoom.max);
  draw();
}

function setZoom(value, clientX, clientY) {
  if (!frame || gesture) return;
  const previous = canvas.getBoundingClientRect();
  const viewport = canvasWrap.getBoundingClientRect();
  const x = Math.max(0, Math.min(previous.width, (clientX ?? (viewport.left + viewport.width/2)) - previous.left));
  const y = Math.max(0, Math.min(previous.height, (clientY ?? (viewport.top + viewport.height/2)) - previous.top));
  zoom.value = Math.max(Number(zoom.min), Math.min(Number(zoom.max), value));
  sizeCanvas();
  const next = canvas.getBoundingClientRect();
  canvasWrap.scrollTo(canvasWrap.scrollLeft + x * (next.width / previous.width - 1),
    canvasWrap.scrollTop + y * (next.height / previous.height - 1));
}

function normalize(points) { return points.map(([x, y]) => [x / canvas.width, y / canvas.height]); }
function pixels(points) { return points.map(([x, y]) => [x * canvas.width, y * canvas.height]); }
function midpoint(a, b) { return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]; }
function distance(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1]); }
function subtraction(a, b) { return [a[0] - b[0], a[1] - b[1]]; }
function visualPixels() { return canvas.width / Math.max(canvas.getBoundingClientRect().width, 1); }

function geometry(box) {
  const [a, b, c, d] = pixels(box.points);
  const [dx, dy] = subtraction(b, a);
  return {center: midpoint(a, c), width: distance(a, b), height: distance(a, d), angle: Math.atan2(dy, dx)};
}

function rectangle({center, width, height, angle}) {
  const cos = Math.cos(angle), sin = Math.sin(angle);
  return [[-width/2, -height/2], [width/2, -height/2], [width/2, height/2], [-width/2, height/2]]
    .map(([x, y]) => [center[0] + x*cos - y*sin, center[1] + x*sin + y*cos]);
}

function clampToImage([x, y]) {
  return [Math.max(0, Math.min(canvas.width, x)), Math.max(0, Math.min(canvas.height, y))];
}

function fits(points) {
  return points.every(([x, y]) => x >= 0 && x <= canvas.width && y >= 0 && y <= canvas.height);
}

function adjustedPoints(points) {
  const xs = points.map(([x]) => x), ys = points.map(([, y]) => y);
  if (Math.max(...xs) - Math.min(...xs) > canvas.width || Math.max(...ys) - Math.min(...ys) > canvas.height) return null;
  const dx = Math.max(-Math.min(...xs), Math.min(0, canvas.width - Math.max(...xs)));
  const dy = Math.max(-Math.min(...ys), Math.min(0, canvas.height - Math.max(...ys)));
  return points.map(([x, y]) => clampToImage([x + dx, y + dy]));
}

function rotationHandle(points) {
  const top = midpoint(points[0], points[1]);
  const center = midpoint(points[0], points[2]);
  const [dx, dy] = subtraction(top, center);
  const length = Math.hypot(dx, dy) || 1;
  const step = 23 * visualPixels();
  return [top[0] + dx/length*step, top[1] + dy/length*step];
}

function strokeBox(points, ink, label, emphasis = false, dashed = false) {
  const scale = visualPixels();
  context.strokeStyle = ink;
  context.fillStyle = ink;
  context.lineWidth = (emphasis ? 2.8 : 1.45) * scale;
  context.setLineDash(dashed ? [4*scale, 3*scale] : []);
  context.beginPath();
  for (const [i, point] of points.entries()) {
    if (i === 0) context.moveTo(...point); else context.lineTo(...point);
  }
  context.closePath();
  context.stroke();
  context.setLineDash([]);
  if (!label) return;
  const [x, y] = points.reduce((best, point) => point[1] < best[1] ? point : best);
  const fontSize = 14 * scale, padding = 2 * scale;
  context.font = fontSize + 'px sans-serif';
  const textX = Math.max(padding, Math.min(x, canvas.width - context.measureText(label).width - padding));
  const textY = Math.max(fontSize + padding, y - padding);
  context.strokeStyle = '#0b1717';
  context.lineWidth = 3*scale;
  context.strokeText(label, textX, textY);
  context.fillText(label, textX, textY);
}

function drawHandles(points) {
  const scale = visualPixels();
  const handle = rotationHandle(points);
  context.lineWidth = 1.4*scale;
  context.strokeStyle = colors.selected;
  context.fillStyle = colors.selected;
  context.beginPath();
  context.moveTo(...midpoint(points[0], points[1]));
  context.lineTo(...handle);
  context.stroke();
  context.beginPath();
  context.arc(...handle, 4.4*scale, 0, 2*Math.PI);
  context.fill();
  context.strokeStyle = '#0b1717';
  for (const point of points) {
    context.beginPath();
    context.arc(...point, 6*scale, 0, 2*Math.PI);
    context.fill();
    context.stroke();
  }
}

function pointer(event) {
  const bounds = canvas.getBoundingClientRect();
  return [(event.clientX - bounds.left)*canvas.width/bounds.width, (event.clientY - bounds.top)*canvas.height/bounds.height];
}

function inside(point, polygon) {
  let result = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [x, y] = polygon[i], [xx, yy] = polygon[j];
    if ((y > point[1]) !== (yy > point[1]) && point[0] < (xx - x)*(point[1] - y)/(yy - y) + x) result = !result;
  }
  return result;
}

function boxHandle(point, points) {
  const scale = visualPixels();
  if (distance(point, rotationHandle(points)) < 9*scale) return {kind: 'rotate'};
  const distances = points.map(position => distance(point, position));
  const nearest = Math.min(...distances);
  if (nearest < 14*scale) {
    const corner = distances.indexOf(nearest);
    return {kind: 'resize', corner, opposite: points[(corner+2)%4]};
  }
  return null;
}

function updateCursor(point) {
  if (gesture?.kind === 'pan') {
    canvas.style.cursor = 'grabbing';
    return;
  }
  if (mode === 'draw') {
    canvas.style.cursor = 'crosshair';
    return;
  }
  const current = selectedBox();
  const points = point && current && isVisible(current) ? pixels(current.points) : null;
  const moving = gesture?.kind === 'move'
    || (!gesture && points && inside(point, points) && !boxHandle(point, points));
  canvas.style.cursor = moving ? 'move' : 'default';
}

function chooseGesture(point) {
  if (mode === 'draw') return {kind: 'draw', start: point};

  const current = selectedBox();
  let hit;
  if (current && isVisible(current)) {
    const points = pixels(current.points);
    const handle = boxHandle(point, points);
    if (handle) return {...handle, start: point, box: current, geometry: geometry(current)};
    if (inside(point, points)) hit = current;
  }
  hit ??= frame.boxes.findLast(box => isVisible(box) && inside(point, pixels(box.points)));
  setSelection(hit?.id);
  if (!hit) return null;
  return {kind: 'move', start: point, box: hit, points: rectangle(geometry(hit))};
}

function draftRectangle(start, point) {
  const [startX, startY] = clampToImage(start), [x, y] = clampToImage(point);
  const right = Math.max(startX, x), left = Math.min(startX, x);
  const bottom = Math.max(startY, y), top = Math.min(startY, y);
  return [[left, top], [right, top], [right, bottom], [left, bottom]];
}

function imageEdgeFraction([x, y], [dx, dy]) {
  const horizontal = dx > 0 ? (canvas.width - x)/dx : dx < 0 ? -x/dx : 1;
  const vertical = dy > 0 ? (canvas.height - y)/dy : dy < 0 ? -y/dy : 1;
  return Math.min(1, horizontal, vertical);
}

function resized(operation, [dx, dy]) {
  const {geometry: initial, opposite, corner} = operation;
  const c = Math.cos(initial.angle), s = Math.sin(initial.angle);
  const widthSign = corner === 0 || corner === 3 ? -1 : 1;
  const heightSign = corner < 2 ? -1 : 1;
  const u = [c*widthSign, s*widthSign], v = [-s*heightSign, c*heightSign];
  const minWidth = Math.min(3, initial.width), minHeight = Math.min(3, initial.height);
  const [x, y] = opposite;
  const minimum = adjustedPoints([opposite, [x + u[0]*minWidth, y + u[1]*minWidth],
    [x + u[0]*minWidth + v[0]*minHeight, y + u[1]*minWidth + v[1]*minHeight],
    [x + v[0]*minHeight, y + v[1]*minHeight]]);
  if (!minimum) return null;
  const [anchor, widthCorner, draggedCorner, heightCorner] = minimum;
  let growWidth = Math.max(0, initial.width + dx*u[0] + dy*u[1] - minWidth);
  let growHeight = Math.max(0, initial.height + dx*v[0] + dy*v[1] - minHeight);

  // Limit each side first so a corner can still slide along a straight image edge.
  const widthDelta = u.map(value => value * growWidth);
  const heightDelta = v.map(value => value * growHeight);
  growWidth *= Math.min(imageEdgeFraction(widthCorner, widthDelta), imageEdgeFraction(draggedCorner, widthDelta));
  growHeight *= Math.min(imageEdgeFraction(heightCorner, heightDelta), imageEdgeFraction(draggedCorner, heightDelta));
  const fraction = imageEdgeFraction(draggedCorner,
    u.map((value, axis) => value*growWidth + v[axis]*growHeight));
  const width = minWidth + growWidth*fraction, height = minHeight + growHeight*fraction;
  const center = anchor.map((value, axis) => value + (u[axis]*width + v[axis]*height)/2);
  return rectangle({center, width, height, angle: initial.angle}).map(clampToImage);
}

function alterGesture(operation, point) {
  const delta = subtraction(point, operation.start);
  if (operation.kind === 'move') return adjustedPoints(operation.points.map(([x,y]) => [x+delta[0],y+delta[1]]));
  if (operation.kind === 'resize') return resized(operation, delta);
  const base = operation.geometry;
  const [tx, ty] = subtraction(operation.start, base.center);
  const [x, y] = subtraction(point, base.center);
  return adjustedPoints(rectangle({...base, angle: base.angle + Math.atan2(y,x) - Math.atan2(ty,tx)}));
}

canvas.addEventListener('pointerdown', event => {
  if (busy || gesture || !image || (event.button !== 0 && event.button !== 2)) return;
  const point = pointer(event);
  if (event.button === 2) {
    gesture = {kind: 'pan', x: event.clientX, y: event.clientY,
      left: canvasWrap.scrollLeft, top: canvasWrap.scrollTop};
  } else {
    gesture = chooseGesture(point);
  }
  updateCursor(point);
  if (gesture) {
    gesture.original = gesture.box?.points;
    canvas.setPointerCapture(event.pointerId);
  }
});

canvas.addEventListener('pointermove', event => {
  const point = pointer(event);
  if (gesture?.kind === 'pan') {
    if (event.buttons & 2) {
      canvasWrap.scrollTo(gesture.left + gesture.x - event.clientX, gesture.top + gesture.y - event.clientY);
    } else {
      gesture = undefined;
      updateCursor(point);
    }
    return;
  }
  updateCursor(point);
  if (!gesture) return;
  if (!gesture.changed && distance(point, gesture.start) < 2*visualPixels()) return;
  if (gesture.kind === 'draw') {
    drawing = draftRectangle(gesture.start, point);
  } else {
    const points = alterGesture(gesture, point);
    if (!points || !fits(points)) return;
    gesture.box.points = normalize(points);
  }
  gesture.changed = true;
  if (gesture.kind === 'draw') draw();
  else updateRegions(false);
});

function classKind(subtype) {
  return materialClasses.has(subtype) ? 'material' : 'foreign';
}

canvas.addEventListener('pointerup', event => {
  if (!gesture) return;
  if (gesture.kind === 'pan') {
    gesture = undefined;
    updateCursor(pointer(event));
    return;
  }
  if (gesture.kind === 'draw') {
    const points = draftRectangle(gesture.start, pointer(event));
    const min = 5 * visualPixels();
    if (distance(points[0], points[1]) >= min && distance(points[0], points[3]) >= min) {
      const subtype = byId('draw-class').value;
      const kind = classKind(subtype);
      const box = {id: crypto.randomUUID(), kind, subtype, points: normalize(points)};
      frame.boxes.push(box);
      selected = box.id;
      setMode('select');
      editedKinds(kind);
      message('');
    } else {
      message('画框宽和高至少需要 5 个屏幕像素。请拖出一个框，细小目标可放大后再画。', true);
    }
  } else if (gesture.changed) {
    editedKinds(gesture.box.kind);
  }
  gesture = drawing = undefined;
  updateRegions();
  updateCursor(pointer(event));
});

canvas.addEventListener('pointercancel', () => {
  if (gesture?.box) gesture.box.points = gesture.original;
  gesture = drawing = undefined;
  updateRegions();
  updateCursor();
});

canvas.addEventListener('lostpointercapture', () => {
  if (gesture?.kind !== 'pan') return;
  gesture = undefined;
  updateCursor();
});
canvas.addEventListener('contextmenu', event => event.preventDefault());
canvas.addEventListener('pointerleave', () => updateCursor());

byId('box-class').addEventListener('change', event => {
  const box = selectedBox();
  if (busy || !box || !isVisible(box)) return;
  const previous = box.kind;
  box.subtype = event.target.value;
  box.kind = classKind(box.subtype);
  editedKinds(previous, box.kind);
  updateRegions();
});

byId('box-angle').addEventListener('change', event => {
  const box = selectedBox(), angle = event.target.valueAsNumber;
  if (busy || !box || !isVisible(box) || !Number.isFinite(angle)) return;
  const points = adjustedPoints(rectangle({...geometry(box), angle: angle * Math.PI/180}));
  if (!points) {
    message('该角度下的框超出图像，请缩小后再调整。', true);
    renderLabels();
    return;
  }
  box.points = normalize(points);
  editedKinds(box.kind);
  updateRegions();
});

function deleteBox() {
  const box = selectedBox();
  if (busy || gesture || !box || !isVisible(box)) return;
  frame.boxes = frame.boxes.filter(item => item.id !== box.id);
  selected = undefined;
  updateCursor();
  editedKinds(box.kind);
  updateRegions();
}

for (const control of ['foreign-reviewed', 'material-reviewed', 'negative']) {
  byId(control).addEventListener('change', event => {
    const checked = event.target.checked;
    frame[control.replace('-', '_')] = checked;
    if (control === 'negative') {
      frame.foreign_reviewed = checked;
      frame.material_reviewed = false;
      frame.verdict = '';
    } else if (control === 'foreign-reviewed' && !checked) {
      frame.material_reviewed = false;
    }
    syncStatus();
    markDirty();
  });
}
byId('note').addEventListener('input', event => {
  frame.note = event.target.value;
  markDirty();
});

byId('delete-box').addEventListener('click', deleteBox);
byId('select-tool').addEventListener('click', () => setMode('select'));
byId('draw-tool').addEventListener('click', () => setMode('draw'));
byId('save').addEventListener('click', () => run(saveFrame));
byId('save-next').addEventListener('click', () => run(async () => {
  const next = neighbor(1);
  await saveFrame();
  if (next) await showFrame(next.id);
}));
byId('batch').addEventListener('change', async event => {
  const requested = event.target.value;
  await run(() => openBatch(requested, frame?.id));
  byId('batch').value = batch;
});
byId('filter').addEventListener('change', () => {
  renderFrames();
  renderControls();
});
byId('frame-search').addEventListener('input', () => {
  renderFrames();
  renderControls();
});
byId('frame-search').addEventListener('keydown', event => {
  if (event.key !== 'Enter' || event.isComposing) return;
  const item = shownFrames()[0];
  if (item) navigate(item.id);
});
byId('prev').addEventListener('click', () => { const item = neighbor(-1); if (item) navigate(item.id); });
byId('next').addEventListener('click', () => { const item = neighbor(1); if (item) navigate(item.id); });
for (const control of ['region', 'materials']) {
  byId(control).addEventListener('change', () => {
    renderLabels();
    draw();
  });
}
byId('export').addEventListener('click', () => run(async () => {
  if (dirty) await saveFrame();
  const {exportable} = await api('/api/frames?batch=' + encodeURIComponent(batch));
  byId('export-count').textContent = exportable > 0
    ? '全部批次共有 ' + exportable + ' 张符合导出条件的图。'
    : '请先确认无异物，或完成阳性图的全图异物和 0.5L 内物料复核。';
  byId('export-download').hidden = exportable === 0;
  byId('export-dialog').showModal();
}));
byId('export-download').addEventListener('click', () => byId('export-dialog').close());
zoom.addEventListener('input', () => setZoom(Number(zoom.value)));
byId('zoom-out').addEventListener('click', () => setZoom(Number(zoom.value) - Number(zoom.step)));
byId('zoom-in').addEventListener('click', () => setZoom(Number(zoom.value) + Number(zoom.step)));
byId('zoom-fit').addEventListener('click', () => setZoom(1));
canvasWrap.addEventListener('wheel', event => {
  if (!frame || !event.ctrlKey || !event.deltaY) return;
  event.preventDefault();
  setZoom(Number(zoom.value) - Math.sign(event.deltaY)*Number(zoom.step), event.clientX, event.clientY);
}, {passive: false});
new ResizeObserver(sizeCanvas).observe(canvasWrap);

document.addEventListener('keydown', event => {
  if (!frame || busy || gesture || event.isComposing || byId('export-dialog').open) return;
  const key = event.key.toLowerCase();
  if ((event.ctrlKey || event.metaKey) && key === 's') {
    event.preventDefault();
    run(saveFrame);
    return;
  }
  if (event.target.matches('input, select, textarea, [contenteditable]') || event.altKey || event.metaKey || event.ctrlKey) return;
  if (key === 'b' || key === 'escape') setMode('select');
  if (key === 'n') setMode('draw');
  if (key === 'delete') deleteBox();
});
window.addEventListener('beforeunload', event => {
  if (!dirty) return;
  event.preventDefault();
  event.returnValue = '';
});

async function autosave() {
  if (!dirty || busy || gesture || pendingSave) return;
  try {
    await saveFrame(true);
  } catch (error) {
    byId('save-state').textContent = '自动保存失败：' + error.message;
  }
}

setInterval(autosave, 10000);

addOptions(byId('draw-class'), Object.entries(classes));
addOptions(byId('box-class'), Object.entries(classes));
const params = new URLSearchParams(location.search);
run(() => openBatch(params.get('batch') || 'pilot', params.get('image')));
