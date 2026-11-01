/* Molecule builder (Avogadro-style): click-to-build in 3D, automatic hydrogens,
   in-browser geometry clean-up and server-side MMFF94/UFF optimisation. */
(function () {
  const { $, $$, api, esc, icon, fmt, toast, saveItem, handoff, copy, download, modal } = SB;

  // ── Element data ──────────────────────────────────────────────────────────
  const EL = {
    H: { r: 0.31, m: 1.008, c: '#e5e7eb', val: [1] }, B: { r: 0.84, m: 10.81, c: '#fca5a5', val: [3] },
    C: { r: 0.76, m: 12.011, c: '#9ca3af', val: [4] }, N: { r: 0.71, m: 14.007, c: '#60a5fa', val: [3, 5] },
    O: { r: 0.66, m: 15.999, c: '#f87171', val: [2] }, F: { r: 0.57, m: 18.998, c: '#86efac', val: [1] },
    Si: { r: 1.11, m: 28.085, c: '#fcd34d', val: [4] }, P: { r: 1.07, m: 30.974, c: '#fb923c', val: [3, 5] },
    S: { r: 1.05, m: 32.06, c: '#facc15', val: [2, 4, 6] }, Cl: { r: 1.02, m: 35.45, c: '#4ade80', val: [1] },
    Br: { r: 1.20, m: 79.904, c: '#c2410c', val: [1] }, I: { r: 1.39, m: 126.9, c: '#a855f7', val: [1] },
  };
  const PALETTE = ['C', 'H', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'B', 'Si'];
  const KEYS = { c: 'C', h: 'H', n: 'N', o: 'O', s: 'S', p: 'P', f: 'F', l: 'Cl', b: 'Br', i: 'I' };
  const TOOLS = [
    ['rotate', 'rotate', 'Navigate', 'Drag to rotate, scroll to zoom, right-drag to pan.'],
    ['draw', 'pen', 'Draw', 'Click an atom to grow a bonded atom of the selected element. Clicking a hydrogen replaces it.'],
    ['bond', 'link', 'Bond', 'Click two atoms: bonds them, or cycles the order (1→2→3) if they are already bonded.'],
    ['delete', 'eraser', 'Delete', 'Click an atom to delete it and its hydrogens.'],
    ['charge', 'charge', 'Charge', 'Click an atom to cycle its formal charge 0 → +1 → −1.'],
    ['measure', 'ruler', 'Measure', 'Pick 2, 3 or 4 atoms for distance, angle or dihedral.'],
  ];
  const TEMPLATES = [['Benzene', 'c1ccccc1'], ['Cyclohexane', 'C1CCCCC1'], ['Pyridine', 'c1ccncc1'], ['Indole', 'c1ccc2[nH]ccc2c1'], ['Naphthalene', 'c1ccc2ccccc2c1'], ['Glycine', 'NCC(=O)O'], ['Aspirin', 'CC(=O)Oc1ccccc1C(=O)O'], ['Caffeine', 'Cn1cnc2c1c(=O)n(C)c(=O)n2C'], ['Ibuprofen', 'CC(C)Cc1ccc(C(C)C(=O)O)cc1'], ['Paracetamol', 'CC(=O)Nc1ccc(O)cc1']];

  // ── State ─────────────────────────────────────────────────────────────────
  let mol = { atoms: [], bonds: [] };
  let undoStack = [], redoStack = [];
  let tool = 'draw', element = 'C', bondOrder = 1, style = 'ballstick';
  let pending = null, picks = [], measures = [];
  let viewer = null, energyTrace = [], lastEnergy = null, busy = false;

  // ── Vector helpers ────────────────────────────────────────────────────────
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const len = (a) => Math.hypot(a[0], a[1], a[2]);
  const norm = (a) => { const l = len(a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
  const P = (i) => [mol.atoms[i].x, mol.atoms[i].y, mol.atoms[i].z];
  const setP = (i, p) => { mol.atoms[i].x = p[0]; mol.atoms[i].y = p[1]; mol.atoms[i].z = p[2]; };
  const rand = () => norm([Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5]);
  function perp(v) { const t = Math.abs(v[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0]; return norm(cross(v, t)); }

  // ── Topology helpers ──────────────────────────────────────────────────────
  const bondsOf = (i) => mol.bonds.filter((b) => b.a === i || b.b === i);
  const other = (b, i) => (b.a === i ? b.b : b.a);
  const neighbors = (i) => bondsOf(i).map((b) => other(b, i));
  const findBond = (i, j) => mol.bonds.find((b) => (b.a === i && b.b === j) || (b.a === j && b.b === i));
  const isH = (i) => mol.atoms[i].el === 'H';
  function heavyValence(i) { return bondsOf(i).filter((b) => !isH(other(b, i))).reduce((s, b) => s + b.order, 0); }
  function allowedValences(i) {
    const a = mol.atoms[i], base = (EL[a.el] || EL.C).val, q = a.charge || 0;
    if (!q) return base;
    if (a.el === 'C') return [3];
    if (a.el === 'B') return q < 0 ? [4] : [2];
    if (['N', 'P', 'O', 'S'].includes(a.el)) return base.map((v) => v + q).filter((v) => v >= 0);
    return base.map((v) => Math.max(0, v - Math.abs(q)));
  }
  function targetH(i) {
    const hv = heavyValence(i), vals = allowedValences(i);
    const v = vals.find((x) => x >= hv);
    return v === undefined ? 0 : v - hv;
  }
  function overValent(i) { const vals = allowedValences(i); return heavyValence(i) + neighbors(i).filter(isH).length > Math.max(...vals); }
  function hybrid(i) {
    const bs = bondsOf(i);
    const doubles = bs.filter((b) => b.order === 2).length, triple = bs.some((b) => b.order === 3);
    if (triple || doubles >= 2) return 1;
    if (doubles === 1) return 2;
    const a = mol.atoms[i];
    if ((a.el === 'N' || a.el === 'O') && bs.some((b) => { const j = other(b, i); return bondsOf(j).some((bb) => bb.order === 2); }) && bs.length <= 3) return 2; // conjugated amide/ester-like
    return 3;
  }
  function bondLength(i, j, order = 1) {
    const f = order === 3 ? 0.78 : order === 2 ? 0.87 : 1;
    return ((EL[mol.atoms[i].el] || EL.C).r + (EL[mol.atoms[j].el] || EL.C).r) * f;
  }

  function removeAtoms(set) {
    const map = new Map(); const atoms = [];
    mol.atoms.forEach((a, i) => { if (!set.has(i)) { map.set(i, atoms.length); atoms.push(a); } });
    mol.bonds = mol.bonds.filter((b) => !set.has(b.a) && !set.has(b.b)).map((b) => ({ a: map.get(b.a), b: map.get(b.b), order: b.order }));
    mol.atoms = atoms;
    return map;
  }

  /** Direction for a new substituent on atom i (away from existing bonds). */
  function freeDirection(i) {
    const nb = neighbors(i);
    if (!nb.length) return rand();
    const vs = nb.map((j) => norm(sub(P(j), P(i))));
    let s = vs.reduce((acc, v) => add(acc, v), [0, 0, 0]);
    if (nb.length === 1) {
      const angle = (hybrid(i) === 1 ? 180 : hybrid(i) === 2 ? 120 : 109.5) * Math.PI / 180;
      const u = vs[0];
      let p = perp(u);
      // keep new atoms roughly in the plane of a grand-neighbour for zig-zag chains
      const gn = neighbors(nb[0]).filter((k) => k !== i);
      if (gn.length) { const w = sub(P(gn[0]), P(nb[0])); const pp = sub(w, mul(u, dot(w, u))); if (len(pp) > 0.1) p = mul(norm(pp), -1); }
      return norm(add(mul(u, Math.cos(angle)), mul(p, Math.sin(angle))));
    }
    if (len(s) < 0.15) return perp(vs[0]);
    return norm(add(mul(s, -1), mul(rand(), 0.15)));
  }

  /** Add/remove hydrogens so atom i satisfies its valence. */
  function adjustH(i) {
    if (i === undefined || i >= mol.atoms.length || isH(i)) return;
    const want = targetH(i);
    let hs = neighbors(i).filter(isH);
    if (hs.length > want) {
      const map = removeAtoms(new Set(hs.slice(0, hs.length - want)));
      return map.get(i);
    }
    for (let k = hs.length; k < want; k++) {
      const d = freeDirection(i);
      mol.atoms.push({ el: 'H', x: 0, y: 0, z: 0, charge: 0 });
      const h = mol.atoms.length - 1;
      setP(h, add(P(i), mul(d, bondLength(i, h))));
      mol.bonds.push({ a: i, b: h, order: 1 });
    }
    return i;
  }

  // ── In-browser force field (fast clean-up, not a substitute for MMFF94) ───
  function relax(iterations = 250) {
    const n = mol.atoms.length; if (n < 2) return;
    const adj = Array.from({ length: n }, () => new Set());
    mol.bonds.forEach((b) => { adj[b.a].add(b.b); adj[b.b].add(b.a); });
    const springs = mol.bonds.map((b) => [b.a, b.b, bondLength(b.a, b.b, b.order), 1.0]);
    const near = Array.from({ length: n }, (_, i) => new Set([i, ...adj[i]]));
    for (let j = 0; j < n; j++) {
      const nb = [...adj[j]]; const hyb = hybrid(j);
      const th = (hyb === 1 ? 180 : hyb === 2 ? 120 : 109.47) * Math.PI / 180;
      for (let x = 0; x < nb.length; x++) for (let y = x + 1; y < nb.length; y++) {
        const i = nb[x], k = nb[y];
        const r1 = bondLength(i, j, (findBond(i, j) || {}).order || 1), r2 = bondLength(k, j, (findBond(k, j) || {}).order || 1);
        springs.push([i, k, Math.sqrt(r1 * r1 + r2 * r2 - 2 * r1 * r2 * Math.cos(th)), 0.45]);
        near[i].add(k); near[k].add(i);
      }
    }
    const pos = mol.atoms.map((a) => [a.x, a.y, a.z]);
    for (let it = 0; it < iterations; it++) {
      const g = pos.map(() => [0, 0, 0]);
      for (const [i, j, r0, k] of springs) {
        const d = sub(pos[i], pos[j]); const r = len(d) || 1e-3; const f = k * (r - r0) / r;
        for (let c = 0; c < 3; c++) { g[i][c] += f * d[c]; g[j][c] -= f * d[c]; }
      }
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
        if (near[i].has(j)) continue;
        const d = sub(pos[i], pos[j]); const r = len(d) || 1e-3;
        const hi = mol.atoms[i].el === 'H', hj = mol.atoms[j].el === 'H';
        const rmin = hi && hj ? 2.1 : hi || hj ? 2.45 : 2.9;
        if (r < rmin) { const f = -0.25 * (rmin - r) / r; for (let c = 0; c < 3; c++) { g[i][c] += f * d[c]; g[j][c] -= f * d[c]; } }
      }
      for (let i = 0; i < n; i++) {
        const step = mul(g[i], -0.35); const l = len(step);
        const s = l > 0.15 ? mul(step, 0.15 / l) : step;
        pos[i] = add(pos[i], s);
      }
    }
    pos.forEach((p, i) => setP(i, p));
  }

  // ── History ───────────────────────────────────────────────────────────────
  function snapshot() { undoStack.push(JSON.stringify(mol)); if (undoStack.length > 80) undoStack.shift(); redoStack = []; }
  function undo() { if (!undoStack.length) return; redoStack.push(JSON.stringify(mol)); mol = JSON.parse(undoStack.pop()); resetPicks(); render(true); }
  function redo() { if (!redoStack.length) return; undoStack.push(JSON.stringify(mol)); mol = JSON.parse(redoStack.pop()); resetPicks(); render(true); }
  function resetPicks() { pending = null; picks = []; }

  function afterEdit(touched) {
    touched.filter((i) => i !== undefined).forEach((i) => adjustH(i));
    if ($('#auto-clean').checked) relax();
    lastEnergy = null; energyTrace = []; measures = [];
    render();
  }

  // ── Editing operations ────────────────────────────────────────────────────
  function grow(i) {
    snapshot();
    let target = i;
    if (!isH(i)) {
      const h = neighbors(i).find(isH);
      if (h !== undefined) target = h;
      else {
        const d = freeDirection(i);
        mol.atoms.push({ el: element, x: 0, y: 0, z: 0, charge: 0 });
        const k = mol.atoms.length - 1;
        mol.bonds.push({ a: i, b: k, order: bondOrder });
        setP(k, add(P(i), mul(d, bondLength(i, k, bondOrder))));
        afterEdit([i, k]);
        return;
      }
    }
    // Replace hydrogen `target` with the selected element.
    const parent = neighbors(target)[0];
    mol.atoms[target].el = element;
    if (parent !== undefined) {
      const b = findBond(parent, target); b.order = bondOrder;
      const d = norm(sub(P(target), P(parent)));
      setP(target, add(P(parent), mul(d, bondLength(parent, target, bondOrder))));
      // adjusting the parent may remove atoms and re-index; resolve indices by object identity
      const tObj = mol.atoms[target], pObj = mol.atoms[parent];
      adjustH(parent);
      afterEdit([mol.atoms.indexOf(tObj), mol.atoms.indexOf(pObj)]);
    } else afterEdit([target]);
  }

  function bondClick(i) {
    if (isH(i)) return toast('Pick heavy atoms to make bonds', 'info', 2000);
    if (pending === null) { pending = i; render(); return; }
    if (pending === i) { pending = null; render(); return; }
    const a = pending; pending = null;
    snapshot();
    const b = findBond(a, i);
    const objA = mol.atoms[a], objB = mol.atoms[i];
    if (b) b.order = b.order >= 3 ? 1 : b.order + 1;
    else mol.bonds.push({ a, b: i, order: 1 });
    adjustH(mol.atoms.indexOf(objA)); adjustH(mol.atoms.indexOf(objB));
    const ia = mol.atoms.indexOf(objA), ib = mol.atoms.indexOf(objB);
    if (overValent(ia) || overValent(ib)) { mol = JSON.parse(undoStack.pop()); toast('That bond would exceed the valence of an atom', 'error'); render(); return; }
    afterEdit([ia, ib]);
  }

  function deleteAtom(i) {
    snapshot();
    const wasH = isH(i);
    const nbObjs = neighbors(i).filter((j) => !isH(j)).map((j) => mol.atoms[j]);
    const kill = new Set([i]);
    if (!wasH) neighbors(i).filter(isH).forEach((h) => kill.add(h));
    removeAtoms(kill);
    measures = []; resetPicks();
    // Deleting a hydrogen leaves an open valence on purpose (e.g. to form a radical site); heavy-atom deletes re-fill neighbours.
    if (wasH) { render(); return; }
    afterEdit(nbObjs.map((o) => mol.atoms.indexOf(o)));
  }

  function chargeClick(i) {
    if (isH(i)) return;
    snapshot();
    const a = mol.atoms[i]; a.charge = a.charge === 0 ? 1 : a.charge === 1 ? -1 : 0;
    afterEdit([i]);
  }

  function measureClick(i) {
    picks.push(i);
    if (picks.length >= 2) {
      const ps = picks.map(P);
      let m = null;
      if (picks.length === 2) m = { kind: 'Distance', value: len(sub(ps[0], ps[1])), unit: 'Å' };
      if (picks.length === 3) { const u = norm(sub(ps[0], ps[1])), v = norm(sub(ps[2], ps[1])); m = { kind: 'Angle', value: Math.acos(Math.max(-1, Math.min(1, dot(u, v)))) * 180 / Math.PI, unit: '°' }; }
      if (picks.length === 4) {
        const b1 = sub(ps[1], ps[0]), b2 = sub(ps[2], ps[1]), b3 = sub(ps[3], ps[2]);
        const n1 = cross(b1, b2), n2 = cross(b2, b3), m1 = cross(n1, norm(b2));
        m = { kind: 'Dihedral', value: Math.atan2(dot(m1, n2), dot(n1, n2)) * 180 / Math.PI, unit: '°' };
      }
      if (m) { m.atoms = picks.slice(); measures = measures.filter((x) => x.atoms.length !== picks.length || picks.length === 2); measures.push(m); }
      if (picks.length === 4) picks = [];
    }
    render();
  }

  function onAtomClick(atom) {
    const i = atom.index;
    if (i === undefined || i >= mol.atoms.length) return;
    ({ draw: grow, bond: bondClick, delete: deleteAtom, charge: chargeClick, measure: measureClick, rotate: () => {} })[tool](i);
  }

  // ── File formats ──────────────────────────────────────────────────────────
  function toMolBlock() {
    const pad = (s, n) => String(s).padStart(n);
    const L = ['SmileBERTa', '  SmileBERTa builder 3D', '', `${pad(mol.atoms.length, 3)}${pad(mol.bonds.length, 3)}  0  0  0  0  0  0  0  0999 V2000`];
    const chg = { 1: 3, 2: 2, 3: 1, '-1': 5, '-2': 6, '-3': 7 };
    mol.atoms.forEach((a) => L.push(`${a.x.toFixed(4).padStart(10)}${a.y.toFixed(4).padStart(10)}${a.z.toFixed(4).padStart(10)} ${a.el.padEnd(3)} 0${pad(chg[a.charge] || 0, 3)}  0  0  0  0  0  0  0  0  0  0`));
    mol.bonds.forEach((b) => L.push(`${pad(b.a + 1, 3)}${pad(b.b + 1, 3)}${pad(b.order, 3)}  0`));
    const charged = mol.atoms.map((a, i) => [i, a.charge]).filter(([, c]) => c);
    for (let k = 0; k < charged.length; k += 8) { const ch = charged.slice(k, k + 8); L.push(`M  CHG${pad(ch.length, 3)}` + ch.map(([i, c]) => `${pad(i + 1, 4)}${pad(c, 4)}`).join('')); }
    L.push('M  END');
    return L.join('\n') + '\n';
  }
  function toXYZ() { return `${mol.atoms.length}\nSmileBERTa builder\n` + mol.atoms.map((a) => `${a.el.padEnd(2)} ${a.x.toFixed(5).padStart(12)} ${a.y.toFixed(5).padStart(12)} ${a.z.toFixed(5).padStart(12)}`).join('\n') + '\n'; }
  function toPDB() {
    const lines = mol.atoms.map((a, i) => `HETATM${String(i + 1).padStart(5)} ${(a.el + (i + 1)).slice(0, 4).padEnd(4)} UNL A   1    ${a.x.toFixed(3).padStart(8)}${a.y.toFixed(3).padStart(8)}${a.z.toFixed(3).padStart(8)}  1.00  0.00          ${a.el.toUpperCase().padStart(2)}`);
    const con = mol.atoms.map((_, i) => { const nb = neighbors(i); return nb.length ? 'CONECT' + [i, ...nb].map((x) => String(x + 1).padStart(5)).join('') : null; }).filter(Boolean);
    return [...lines, ...con, 'END'].join('\n') + '\n';
  }
  async function toSmiles() {
    try { const R = await SB.loadRDKit(); const m = R.get_mol(toMolBlock(), JSON.stringify({ removeHs: true })); if (!m || !m.is_valid()) return null; const s = m.get_smiles(); m.delete(); return s; } catch (e) { return null; }
  }

  // ── Rendering ─────────────────────────────────────────────────────────────
  const STYLES = {
    ballstick: { stick: { radius: 0.13, colorscheme: 'Jmol' }, sphere: { scale: 0.24, colorscheme: 'Jmol' } },
    stick: { stick: { radius: 0.17, colorscheme: 'Jmol' } },
    sphere: { sphere: { scale: 1.0, colorscheme: 'Jmol' } },
    line: { line: { linewidth: 2, colorscheme: 'Jmol' }, sphere: { scale: 0.1, colorscheme: 'Jmol' } },
  };
  function render(fit = false) {
    if (!viewer) return;
    const view = viewer.getView();
    viewer.removeAllModels(); viewer.removeAllShapes(); viewer.removeAllLabels();
    $('#first-atom').classList.toggle('hidden', mol.atoms.length > 0);
    if (mol.atoms.length) {
      const m = viewer.addModel(toMolBlock(), 'sdf', { keepH: true });
      viewer.setStyle({}, STYLES[style]);
      if (!$('#show-h').checked) viewer.setStyle({ elem: 'H' }, {});
      viewer.setClickable({}, true, (atom) => onAtomClick(atom));
      viewer.setHoverable({}, true, (atom) => showTip(atom), () => hideTip());
      if ($('#show-labels').checked) mol.atoms.forEach((a, i) => { if (a.el !== 'H' || $('#show-h').checked) viewer.addLabel(a.el + (i + 1), { position: { x: a.x, y: a.y, z: a.z }, fontSize: 11, fontColor: 'white', backgroundOpacity: 0.35, backgroundColor: '#0b1020', inFront: true, showBackground: true }); });
      mol.atoms.forEach((a, i) => { if (a.charge) viewer.addLabel(a.charge > 0 ? '+' : '−', { position: { x: a.x + 0.35, y: a.y + 0.35, z: a.z }, fontSize: 14, fontColor: a.charge > 0 ? '#60a5fa' : '#f87171', showBackground: false, inFront: true }); });
      mol.atoms.forEach((a, i) => { if (!isH(i) && overValent(i)) viewer.addSphere({ center: { x: a.x, y: a.y, z: a.z }, radius: 0.6, color: '#ef4444', opacity: 0.35 }); });
      if (pending !== null) { const a = mol.atoms[pending]; viewer.addSphere({ center: { x: a.x, y: a.y, z: a.z }, radius: 0.55, color: '#22d3ee', opacity: 0.45 }); }
      picks.forEach((p) => { const a = mol.atoms[p]; viewer.addSphere({ center: { x: a.x, y: a.y, z: a.z }, radius: 0.5, color: '#fbbf24', opacity: 0.4 }); });
      measures.forEach((ms) => {
        const pts = ms.atoms.map((i) => mol.atoms[i]).filter(Boolean);
        if (pts.length !== ms.atoms.length) return;
        for (let k = 0; k < pts.length - 1; k++) viewer.addCylinder({ start: pts[k], end: pts[k + 1], radius: 0.03, color: '#fbbf24', dashed: true, fromCap: 1, toCap: 1 });
        const c = pts.reduce((s, p) => ({ x: s.x + p.x / pts.length, y: s.y + p.y / pts.length, z: s.z + p.z / pts.length }), { x: 0, y: 0, z: 0 });
        viewer.addLabel(`${fmt(ms.value, ms.unit === 'Å' ? 3 : 1)} ${ms.unit}`, { position: c, fontSize: 12, fontColor: '#fde68a', backgroundColor: '#1a1405', backgroundOpacity: 0.85, inFront: true });
      });
      void m;
    }
    if (fit || !view || mol.atoms.length <= 1) { viewer.zoomTo(); if (mol.atoms.length < 4) viewer.zoom(0.5); } else viewer.setView(view);
    viewer.render();
    updateInfo();
  }
  const tip = SB.h('<div class="tooltip-3d hidden"></div>'); document.body.appendChild(tip);
  let mouse = { x: 0, y: 0 }; document.addEventListener('mousemove', (e) => { mouse = { x: e.clientX, y: e.clientY }; tip.style.left = mouse.x + 'px'; tip.style.top = mouse.y + 'px'; });
  function showTip(atom) {
    const i = atom.index; if (i === undefined || !mol.atoms[i]) return;
    const a = mol.atoms[i];
    tip.textContent = `${a.el}${i + 1}${a.charge ? (a.charge > 0 ? ' +' : ' −') : ''} · ${neighbors(i).length} bonds${isH(i) ? '' : ' · sp' + (hybrid(i) === 3 ? '³' : hybrid(i) === 2 ? '²' : '')}`;
    tip.classList.remove('hidden');
  }
  function hideTip() { tip.classList.add('hidden'); }
  document.addEventListener('mouseover', (e) => { if (!e.target.closest('#viewer')) hideTip(); });

  const updateSmiles = SB.debounce(async () => {
    const s = mol.atoms.length ? await toSmiles() : null;
    const el = $('#prop-smiles'); if (el) el.textContent = s || (mol.atoms.length ? '(invalid)' : '—');
  }, 400);
  function updateInfo() {
    const counts = {}; let mass = 0, q = 0;
    mol.atoms.forEach((a) => { counts[a.el] = (counts[a.el] || 0) + 1; mass += (EL[a.el] || EL.C).m; q += a.charge || 0; });
    const order = ['C', 'H', ...Object.keys(counts).filter((e) => e !== 'C' && e !== 'H').sort()];
    const formula = order.filter((e) => counts[e]).map((e) => e + (counts[e] > 1 ? counts[e] : '')).join('');
    $('#hud-info').innerHTML = mol.atoms.length ? `${esc(formula)} · ${mol.atoms.length} atoms · ${mol.bonds.length} bonds${q ? ` · charge ${q > 0 ? '+' : ''}${q}` : ''}` : 'Empty scene';
    $('#props').innerHTML = `<div>Formula</div><div>${esc(formula || '—')}</div><div>Mol. weight</div><div>${mass ? fmt(mass, 2) + ' Da' : '—'}</div><div>Atoms / bonds</div><div>${mol.atoms.length} / ${mol.bonds.length}</div><div>Net charge</div><div>${q}</div><div>SMILES</div><div id="prop-smiles" style="word-break:break-all;max-width:180px;text-align:right">…</div>`;
    updateSmiles();
    const bad = mol.atoms.some((_, i) => !isH(i) && overValent(i));
    $('#hud-energy').classList.toggle('hidden', lastEnergy === null && !bad);
    $('#hud-energy').innerHTML = bad ? '<span style="color:var(--red)">⚠ Valence error (red atoms)</span>' : lastEnergy !== null ? `E = ${fmt(lastEnergy.e, 3)} kcal/mol · ${esc(lastEnergy.ff)}` : '';
    drawChart();
  }

  function drawChart() {
    const c = $('#energy-chart'); const dpr = devicePixelRatio || 1;
    c.width = c.clientWidth * dpr; c.height = c.clientHeight * dpr;
    const g = c.getContext('2d'); g.scale(dpr, dpr);
    const W = c.clientWidth, H = c.clientHeight;
    g.clearRect(0, 0, W, H);
    if (energyTrace.length < 2) { g.fillStyle = '#4a5368'; g.font = '11px Inter'; g.fillText('Energy trace appears after optimisation', 10, H / 2 + 4); return; }
    const min = Math.min(...energyTrace), max = Math.max(...energyTrace), pad = 10;
    const x = (i) => pad + i / (energyTrace.length - 1) * (W - 2 * pad), y = (v) => pad + (1 - (v - min) / (max - min || 1)) * (H - 2 * pad);
    const grad = g.createLinearGradient(0, 0, W, 0); grad.addColorStop(0, '#3b82f6'); grad.addColorStop(1, '#2dd4bf');
    g.beginPath(); energyTrace.forEach((v, i) => (i ? g.lineTo(x(i), y(v)) : g.moveTo(x(i), y(v)))); g.strokeStyle = grad; g.lineWidth = 2; g.stroke();
    g.lineTo(x(energyTrace.length - 1), H); g.lineTo(x(0), H); g.closePath(); g.fillStyle = 'rgba(34,211,238,.08)'; g.fill();
    g.fillStyle = '#7d879c'; g.font = '10px JetBrains Mono'; g.fillText(fmt(max, 1), 6, 12); g.fillText(fmt(min, 1), 6, H - 4);
  }

  // ── Server operations ─────────────────────────────────────────────────────
  async function optimize() {
    if (!mol.atoms.length || busy) return;
    if (mol.atoms.some((_, i) => !isH(i) && overValent(i))) return toast('Fix the valence errors (red atoms) first', 'error');
    busy = true;
    const btn = $('#optimize'); btn.classList.add('loading'); btn.innerHTML = '<span class="spinner"></span> Optimising…';
    try {
      const ff = $('#ff').value;
      const d = await api('/builder/optimize', { body: { atoms: mol.atoms, bonds: mol.bonds, forcefield: ff, steps: +$('#steps').value, frames: 40 } });
      snapshot();
      energyTrace = [d.energy_initial];
      for (const fr of d.frames) {
        fr.coords.length === mol.atoms.length * 3 && mol.atoms.forEach((a, i) => { a.x = fr.coords[3 * i]; a.y = fr.coords[3 * i + 1]; a.z = fr.coords[3 * i + 2]; });
        energyTrace.push(fr.energy); lastEnergy = { e: fr.energy, ff: d.forcefield };
        render(); await SB.sleep(35);
      }
      mol = { atoms: d.atoms, bonds: d.bonds };
      lastEnergy = { e: d.energy_final, ff: d.forcefield };
      render();
      $('#energy-kv').innerHTML = `<div>Force field</div><div>${esc(d.forcefield)}</div><div>Initial energy</div><div>${fmt(d.energy_initial, 3)}</div><div>Final energy</div><div style="color:var(--green)">${fmt(d.energy_final, 3)}</div><div>ΔE</div><div>${fmt(d.energy_final - d.energy_initial, 3)} kcal/mol</div><div>Converged</div><div>${d.converged ? 'yes' : 'no, run again'} (${d.steps} steps)</div>`;
      if (d.forcefield !== ff) toast(`${ff} has no parameters for this molecule, so UFF was used instead`, 'info');
      if (d.properties) showAnalysis(d.properties);
    } catch (e) { toast(e.message, 'error'); }
    finally { busy = false; btn.classList.remove('loading'); btn.innerHTML = icon('zap') + ' Optimise geometry'; }
  }

  async function loadSmiles(smiles, label) {
    if (!smiles) return;
    try {
      toast('Building 3D structure' + (label ? ' for ' + label : '') + '…', 'info', 1500);
      const d = await api('/builder/from_smiles', { body: { smiles } });
      snapshot(); mol = { atoms: d.atoms, bonds: d.bonds }; resetPicks(); measures = []; energyTrace = []; lastEnergy = null;
      render(true);
      if (d.properties) showAnalysis(d.properties);
    } catch (e) { toast(e.message, 'error'); }
  }

  async function conformers() {
    if (!mol.atoms.length) return;
    const btn = $('#conf-run'); btn.classList.add('loading');
    $('#conf-list').innerHTML = '<div class="skeleton" style="height:120px"></div>';
    try {
      const d = await api('/builder/conformers', { body: { atoms: mol.atoms, bonds: mol.bonds, n: +$('#n-conf').value, forcefield: $('#ff').value } });
      $('#conf-list').innerHTML = d.conformers.map((c, i) => `<div class="list-item conf-item" data-i="${i}"><b style="width:26px">#${i + 1}</b><div class="grow mono" style="font-size:.76rem">${fmt(c.energy, 2)} kcal/mol</div><span class="badge ${i ? '' : 'green'}">+${fmt(c.relative, 2)}</span></div>`).join('') + `<p class="kbd-hint" style="margin:4px 0 0">${d.conformers.length} unique conformers · ${esc(d.forcefield)} · click to view</p>`;
      $$('.conf-item').forEach((el) => el.addEventListener('click', () => {
        $$('.conf-item').forEach((x) => x.classList.toggle('active', x === el));
        const c = d.conformers[+el.dataset.i];
        snapshot(); mol = { atoms: c.atoms, bonds: c.bonds }; lastEnergy = { e: c.energy, ff: d.forcefield }; resetPicks(); render(true);
      }));
    } catch (e) { $('#conf-list').innerHTML = ''; toast(e.message, 'error'); }
    finally { btn.classList.remove('loading'); }
  }

  function showAnalysis(p) {
    $('#analysis').innerHTML = `${SB.druglikeBadges(p)}<div class="kv" style="margin-top:10px"><div>cLogP</div><div>${fmt(p.log_p)}</div><div>TPSA</div><div>${fmt(p.tpsa, 1)} Å²</div><div>HBD / HBA</div><div>${p.hydrogen_bond_donors} / ${p.hydrogen_bond_acceptors}</div><div>Rot. bonds</div><div>${p.rotatable_bonds}</div><div>Rings (arom.)</div><div>${p.rings} (${p.aromatic_rings})</div><div>Fsp³</div><div>${fmt(p.fraction_sp3, 2)}</div></div>`;
  }

  // ── UI wiring ─────────────────────────────────────────────────────────────
  function setTool(t) {
    tool = t; resetPicks();
    $$('#tools .toolbtn').forEach((b) => b.classList.toggle('active', b.dataset.t === t));
    const info = TOOLS.find((x) => x[0] === t);
    $('#tool-hint').textContent = info[3];
    $('#hud-tool').innerHTML = `${icon(info[1])} <b>${info[2]}</b>${t === 'draw' ? ` · ${element}${bondOrder > 1 ? ' · ' + ['', '', 'double', 'triple'][bondOrder] : ''}` : ''}`;
    $('#hud-tool svg') && Object.assign($('#hud-tool svg').style, { width: '14px', height: '14px', verticalAlign: '-2px' });
    render();
  }
  function setElement(e) { element = e; $$('.el-btn').forEach((b) => b.classList.toggle('active', b.dataset.el === e)); $('#place-first').textContent = e; if (tool !== 'draw') setTool('draw'); else setTool('draw'); }
  function setOrder(o) { bondOrder = o; $$('#bond-order button').forEach((b) => b.classList.toggle('active', +b.dataset.o === o)); setTool(tool); }

  function init() {
    if (!window.$3Dmol) return setTimeout(init, 80);
    viewer = $3Dmol.createViewer($('#viewer'), { backgroundColor: '#070a12', antialias: true });
    $('#tools').innerHTML = TOOLS.map(([k, ic, name]) => `<button class="toolbtn" data-t="${k}" title="${name}">${icon(ic)}${name}</button>`).join('');
    $$('#tools .toolbtn').forEach((b) => b.addEventListener('click', () => setTool(b.dataset.t)));
    $('#elements').innerHTML = PALETTE.map((e) => `<button class="el-btn" data-el="${e}" style="--c:${EL[e].c}" title="${e}">${e}</button>`).join('');
    $$('.el-btn').forEach((b) => b.addEventListener('click', () => setElement(b.dataset.el)));
    $$('#bond-order button').forEach((b) => b.addEventListener('click', () => setOrder(+b.dataset.o)));
    $('#templates').innerHTML = TEMPLATES.map(([n, s]) => `<button class="chip" data-s="${esc(s)}">${n}</button>`).join('');
    $$('#templates .chip').forEach((c) => c.addEventListener('click', () => loadSmiles(c.dataset.s, c.textContent)));
    $('#smiles-load').addEventListener('click', () => loadSmiles($('#smiles-in').value.trim()));
    $('#smiles-in').addEventListener('keydown', (e) => { if (e.key === 'Enter') loadSmiles(e.target.value.trim()); });
    $('#undo').innerHTML = icon('undo'); $('#redo').innerHTML = icon('redo');
    $('#undo').addEventListener('click', undo); $('#redo').addEventListener('click', redo);
    $('#clear').addEventListener('click', () => { if (!mol.atoms.length) return; snapshot(); mol = { atoms: [], bonds: [] }; measures = []; resetPicks(); energyTrace = []; lastEnergy = null; $('#energy-kv').innerHTML = ''; $('#analysis').innerHTML = ''; render(true); });
    $$('#style-seg button').forEach((b) => b.addEventListener('click', () => { style = b.dataset.s; $$('#style-seg button').forEach((x) => x.classList.toggle('active', x === b)); render(); }));
    $('#show-h').addEventListener('change', () => render()); $('#show-labels').addEventListener('change', () => render());
    $('#place-first').addEventListener('click', () => { snapshot(); mol.atoms.push({ el: element, x: 0, y: 0, z: 0, charge: 0 }); afterEdit([0]); render(true); });
    $('#optimize').innerHTML = icon('zap') + ' Optimise geometry';
    $('#optimize').addEventListener('click', optimize);
    $('#quick-clean').addEventListener('click', () => { if (!mol.atoms.length) return; snapshot(); relax(600); render(); });
    $('#conf-run').addEventListener('click', conformers);
    $('#analyze').addEventListener('click', async () => {
      const s = await toSmiles(); if (!s) return toast('Molecule is not valid yet', 'error');
      try { const d = await api('/analyze', { body: { smiles: s } }); showAnalysis(d.properties); } catch (e) { toast(e.message, 'error'); }
    });
    $('#save').innerHTML = icon('save') + ' Save';
    $('#save').addEventListener('click', async () => {
      if (!mol.atoms.length) return toast('Nothing to save yet', 'error');
      const s = await toSmiles();
      saveItem('molecule', s ? 'Molecule ' + s.slice(0, 40) : 'Built molecule', { atoms: mol.atoms, bonds: mol.bonds, smiles: s, energy: lastEnergy }, s || '');
    });
    $('#to-dock').innerHTML = icon('target') + ' Dock';
    $('#to-dock').addEventListener('click', async () => {
      if (!mol.atoms.length) return toast('Build a molecule first', 'error');
      handoff('docking', { molblock: toMolBlock(), smiles: await toSmiles(), name: 'Builder molecule' });
    });
    // export
    $$('[data-drop]', $('.app-topbar')).forEach((d) => d.querySelector('button').addEventListener('click', (e) => { e.stopPropagation(); d.classList.toggle('open'); }));
    $$('[data-exp]').forEach((a) => a.addEventListener('click', async (e) => {
      e.preventDefault(); $('#export-drop').classList.remove('open');
      if (!mol.atoms.length) return toast('Nothing to export', 'error');
      const k = a.dataset.exp;
      if (k === 'sdf') download(toMolBlock() + '$$$$\n', 'molecule.sdf', 'chemical/x-mdl-sdfile');
      if (k === 'xyz') download(toXYZ(), 'molecule.xyz', 'chemical/x-xyz');
      if (k === 'pdb') download(toPDB(), 'molecule.pdb', 'chemical/x-pdb');
      if (k === 'smiles') { const s = await toSmiles(); s ? copy(s) : toast('Molecule is not valid yet', 'error'); }
      if (k === 'png') { const a2 = document.createElement('a'); a2.href = viewer.pngURI(); a2.download = 'molecule.png'; a2.click(); }
    }));
    // import
    $('#import').addEventListener('click', () => {
      const inp = SB.h('<input type="file" accept=".sdf,.mol,.xyz,.pdb,.mol2" hidden>'); document.body.appendChild(inp);
      inp.addEventListener('change', async () => {
        const f = inp.files[0]; inp.remove(); if (!f) return;
        const ext = f.name.split('.').pop().toLowerCase(); const text = await SB.readFile(f);
        try {
          viewer.removeAllModels();
          const m = viewer.addModel(text, ext === 'mol' ? 'sdf' : ext, { keepH: true });
          const atoms = m.selectedAtoms({});
          if (atoms.length > 400) throw new Error('Too many atoms for the builder (max 400). Use Structure Studio for proteins.');
          const idx = new Map(atoms.map((a, i) => [a.index, i]));
          const bonds = [];
          atoms.forEach((a, i) => (a.bonds || []).forEach((j, k) => { const jj = idx.get(j); if (jj !== undefined && jj > i) bonds.push({ a: i, b: jj, order: Math.min(3, Math.max(1, Math.round(a.bondOrder ? a.bondOrder[k] : 1))) }); }));
          snapshot();
          mol = { atoms: atoms.map((a) => ({ el: a.elem.length > 1 ? a.elem[0].toUpperCase() + a.elem.slice(1).toLowerCase() : a.elem.toUpperCase(), x: a.x, y: a.y, z: a.z, charge: 0 })), bonds };
          resetPicks(); measures = []; render(true); toast(`Imported ${atoms.length} atoms from ${f.name}`, 'ok');
        } catch (err) { toast(err.message || 'Could not read file', 'error'); render(true); }
      });
      inp.click();
    });
    // keyboard
    document.addEventListener('keydown', (e) => {
      if (e.target.matches('input, textarea, select')) return;
      const k = e.key.toLowerCase();
      if ((e.metaKey || e.ctrlKey) && k === 'z') { e.preventDefault(); e.shiftKey ? redo() : undo(); return; }
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (KEYS[k]) setElement(KEYS[k]);
      else if (['1', '2', '3'].includes(k)) setOrder(+k);
      else if (k === 'escape') { if (pending !== null || picks.length) { resetPicks(); render(); } else setTool('rotate'); }
      else if (k === 'delete' || k === 'backspace') setTool('delete');
      else if (k === 'm') setTool('measure');
    });

    setElement('C'); setTool('draw');
    drawChart();
    // Restore from workspace / hand-off
    (async () => {
      const id = SB.param('item');
      if (id) { try { const it = await SB.loadItem(id); if (it.data && it.data.atoms) { mol = { atoms: it.data.atoms, bonds: it.data.bonds }; render(true); toast('Loaded “' + it.name + '”', 'ok'); return; } } catch (e) { toast(e.message, 'error'); } }
      const ho = SB.takeHandoff('builder');
      if (ho && ho.smiles) return loadSmiles(ho.smiles);
      render(true);
    })();
    addEventListener('resize', () => { viewer.resize(); drawChart(); });
  }
  init();
})();
