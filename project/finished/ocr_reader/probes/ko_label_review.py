#!/usr/bin/env python3
"""A local review page for the Korean ``OCR:`` labels of the two-prompt reader
(``manifest_sfxp_kozh.parquet``: gelnote KO and swaprelax_v5) and the hito v4 self-labels
(``manifest_pseudo_hito_v4.parquet``, one source per side: ``hito_v4_ja`` / ``hito_v4_ko``):
the crop as trained, the label in an editable box, a status per row. Edits stay in the
browser (localStorage) and leave as a TSV through the page's Export button
(``apply_ko_review.py`` takes it).

Every editable row has an ``SFX`` checkbox (exported as the ``sfx`` column;
``apply_ko_review.py`` keeps the marks in ``assets/sfx_marks.tsv``, training does not read
them); a row the ``--flags`` reader read under ``SFX:`` comes up checked. ``m109_sfx`` / ``m109_speech`` are read-only references: a seeded sample of the
Manga109 COO onomatopoeia and the speech text (``manifest.parquet``), to compare against.

Every editable row has a ``+ sub`` button: a second box for the ``OCR2:`` label of the same
crop (furigana, a smaller SFX, a hand-lettered aside; one item per line, reading order),
exported as the ``sub`` column (``-`` = removed). ``furigana`` (``--ruby``,
``furigana_find.py``): speech crops with ruby groups found, the sub box prefilled with the
groups read as kana.

``mainsub_sfx_wide`` / ``mainsub_det`` / ``mainsub_speech`` (``--mainsub``, ``m109_mainsub.py``):
Manga109 crops with the box holding the proposed ``OCR:`` main (the largest text) and the
sub box the proposed ``OCR2:`` sub (the other annotated text in the crop); the meta line
lists each item's glyph size and the text only partly inside the crop (type what is
visible into the sub). Every row with a sub or partly visible text is listed, plus
``--mainsub_main_n`` seeded main-only rows per source. An emptied main box is a crop with
only sub text. The header's ``has`` filter (any source) lists rows by what their boxes
hold now: main only, sub only, both, or partly visible text.
``apply_ko_review.py`` writes these rows to ``assets/m109_mainsub.tsv``.

``linebreak_suspicious`` (``linebreak_auto.py apply``): JA speech rows whose automatic line
breaks missed the gate; the box holds the proposed breaks, the meta line the per-column
read and its distance to the label (largest first). ``ok`` takes the proposal.
``odd_dots``: speech rows with text whose dots ``ellipsis_fold`` left odd, the box holding
``round_dots``' spelling; same ``ok`` rule, split runs first. (``dots_suspicious``, the
dot-only rows ``linebreak_auto.py dots`` could not spell, was reviewed in full on 11-18 and
is no longer listed; ``apply_ko_review.py`` still takes its rows.)

hito rows also show the twin crop (the other edition's lettering of the same region) with
its v4 read, and the read's mean token logprob (``prelabel_v4lp.jsonl``); they sort lowest
logprob first by default.

Every row is listed; the page filters on the longest spaceless Hangul run (the gelnote
teacher's dropped spacing passes the whitespace-blind agreement, ``pseudo_label.py``) and
sorts by it, by "no line break, longest first", or by ``suspicion`` (``suspicion``: the
label's line count against the crop's shape, kana / hanja, long spaceless runs). With
``--flags`` (``ko_read.py`` → ``ko_label_fix.py``) every kozh and hito row also shows a
reader's read and confidence, and the "disagrees" sort lists the rows whose label the read
does not match (whitespace-blind), lowest confidence first. Re-read with each new run so
that sort follows the latest reader.

``doujin_sfx``: the doujin gate's SFX rows (``assets/sfx_labels_sincos_597.tsv``,
``kind_hand == sfx``, ``unchecked`` included) on the frozen 12 % crops
(``eval_data/sincos``); the meta line shows the row's label status, its pre-label
(``text_rec``) and note. ``--doujin_reads`` (``eval_sfx.py``'s
``output/ocr/eval/sfx_<name>.jsonl``) adds that reader's read; its "confidence" is
``eval_sfx``'s sim to the label, and "disagrees" leaves out the rows ``eval_sfx`` scores
exact (its heart / shape folds). ``apply_ko_review.py`` writes these rows back to the TSV.

    python project/finished/ocr_reader/probes/ko_label_review.py --out ~/ko_label_review
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ocr"))
import manga109 as m109  # noqa: E402
from build_sfx_prefix import ellipsis_fold  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
GELNOTE = Path("/media/sorryhyun/new/dataset/gelnote_crops")
SWAPRELAX = Path("/media/sorryhyun/new/dataset/hito_ocr/swap_relax_v5")
HITO = Path("/media/sorryhyun/new/dataset/hito_ocr/ko")
OCR = Path(__file__).resolve().parents[1]
DOUJIN_LABELS = OCR / "assets/sfx_labels_sincos_597.tsv"
DOUJIN_FROZEN = OCR / "eval_data/sincos"
HANGUL = re.compile(r"[가-힣]")
RUN = re.compile(r"[가-힣]+")
KANA = re.compile(r"[\u3040-\u30ff]")
HANJA = re.compile(r"[\u4e00-\u9fff]")
DOTS = re.compile(r"[.．・･]{2,}")
DOT_FREE = re.compile(r"[\s.．・･…‥]")
MIXED = re.compile(r"[.．・･]…|…[.．・･]")
SPLIT = re.compile(r"[.．・･…]\n[.．・･…]")

PAGE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Label review</title>
<style>
:root { --bg:#f6f6f4; --card:#fff; --ink:#1d1d1b; --mute:#77756f; --line:#dddbd5;
        --ok:#2f7d4f; --edit:#2a5db0; --drop:#b0372a; --hl:#ffe08a; }
@media (prefers-color-scheme: dark) { :root { --bg:#1b1b1a; --card:#262624; --ink:#ecebe6;
        --mute:#9c9a93; --line:#3a3936; --hl:#6b5a1c; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
       font:15px/1.5 "Noto Sans CJK KR","Noto Sans KR",system-ui,sans-serif; }
header { position:sticky; top:0; z-index:2; background:var(--bg); border-bottom:1px solid var(--line);
         padding:10px 16px; display:flex; flex-wrap:wrap; gap:10px 18px; align-items:center; }
header h1 { font-size:17px; margin:0 8px 0 0; }
header label { color:var(--mute); font-size:13px; }
select, input[type=number] { font:inherit; padding:2px 6px; background:var(--card); color:var(--ink);
         border:1px solid var(--line); border-radius:4px; }
input[type=number] { width:56px; }
button { font:inherit; font-size:13px; padding:3px 10px; border:1px solid var(--line); border-radius:4px;
         background:var(--card); color:var(--ink); cursor:pointer; }
.stats { color:var(--mute); font-size:13px; }
main { padding:12px 16px 80px; max-width:1200px; margin:0 auto; }
.row { display:grid; grid-template-columns:minmax(0,320px) minmax(0,1fr); gap:14px; background:var(--card);
       border:1px solid var(--line); border-left:4px solid var(--line); border-radius:6px;
       padding:10px; margin-bottom:10px; }
.row.ok { border-left-color:var(--ok); } .row.edit { border-left-color:var(--edit); }
.row.drop { border-left-color:var(--drop); opacity:.6; }
.crop { display:flex; align-items:flex-start; justify-content:center; background:#fff; border-radius:4px;
        min-height:60px; }
.crop img { max-width:100%; max-height:280px; object-fit:contain; }
.twin { display:flex; gap:8px; align-items:flex-start; margin:2px 0 6px; }
.twin img { max-width:140px; max-height:120px; object-fit:contain; background:#fff; border-radius:3px; }
.meta { color:var(--mute); font-size:12px; margin-bottom:4px; word-break:break-all; }
.meta a { color:var(--mute); }
.orig { font-size:14px; margin:2px 0 6px; white-space:pre-wrap; }
.orig mark { background:var(--hl); color:inherit; border-radius:2px; }
textarea { width:100%; font:inherit; font-size:17px; padding:6px 8px; border:1px solid var(--line);
           border-radius:4px; background:var(--card); color:var(--ink); resize:vertical; }
.btns { display:flex; gap:6px; margin-top:6px; flex-wrap:wrap; }
.btns label { font-size:13px; display:flex; gap:4px; align-items:center; margin-left:8px; }
.row.sfx { box-shadow: inset 0 0 0 2px #c77d1a; }
.row.hassub .subta { border-color:#2a7fbf; }
.subbox { margin-top:6px; } .subta { font-size:15px; }
.btns .on.ok { background:var(--ok); color:#fff; } .btns .on.drop { background:var(--drop); color:#fff; }
.pager { display:flex; gap:8px; align-items:center; justify-content:center; margin:14px 0; }
header .pager { margin:0; }
@media (max-width:640px) { .row { grid-template-columns:minmax(0,1fr); } }
</style></head><body>
<header>
  <h1>Label review</h1>
  <label>source <select id="src"></select></label>
  <label>min run <input id="minrun" type="number" min="0" max="20"></label>
  <label>sort <select id="sort"><option value="sus">suspicion</option>
    <option value="lp">lowest logprob first (hito)</option>
    <option value="sfx">SFX-marked only</option>
    <option value="rsfx">__READER__ read under SFX: only</option>
    <option value="nolf">no line break, longest first</option>
    <option value="edge">edge text (label beyond __READER__'s read)</option>
    <option value="dis">__READER__ disagrees, lowest confidence first</option>
    <option value="run">longest run first</option>
    <option value="id">id</option><option value="todo">unreviewed first</option></select></label>
  <label>has <select id="has"><option value="">all</option><option value="main">main only</option>
    <option value="sub">sub only</option><option value="both">main + sub</option>
    <option value="part">partly visible text</option></select></label>
  <label><input type="checkbox" id="donelast"> reviewed last</label>
  <span class="stats" id="stats"></span>
  <span class="stats">edits save as you type · Enter or \n = line break</span>
  <button id="export">Export TSV</button>
  <span class="pager" id="pagerTop"></span>
</header>
<main><div id="list"></div><div class="pager" id="pager"></div></main>
<script>
const DATA = __DATA__;
const MIN_DEFAULT = __MINRUN__;
const SORT_DEFAULT = {"gelnote_ko": "nolf", "swaprelax_v5": "sus", "hito_v4_ja": "lp", "hito_v4_ko": "lp",
                      "linebreak_suspicious": "sus", "odd_dots": "sus", "furigana": "sus", "doujin_sfx": "__DOUJIN_SORT__",
                      "mainsub_sfx_wide": "sus", "mainsub_det": "sus", "mainsub_speech": "sus"};
const PER = 50;
const KEY = "ko_label_review_v1";
let store = {};
try { store = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { store = {}; }
let marks = {};
try { marks = JSON.parse(localStorage.getItem(KEY + ":sfx") || "{}"); } catch (e) { marks = {}; }
// sub (OCR2:) text per row: absent = the row's proposal (x.sub) if any, "" = no sub
let subs = {};
try { subs = JSON.parse(localStorage.getItem(KEY + ":sub") || "{}"); } catch (e) { subs = {}; }
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(store));
  localStorage.setItem(KEY + ":sfx", JSON.stringify(marks));
  localStorage.setItem(KEY + ":sub", JSON.stringify(subs)); } catch (e) {} };
const subOf = (key, x) => key in subs ? subs[key] : (x.sub || "");
const $ = (id) => document.getElementById(id);
// a read from the reader's SFX: pass checks the row's SFX box once; an uncheck is kept as 0
(() => {
  let n = 0;
  for (const [s, rows] of Object.entries(DATA)) for (const x of rows) {
    const k = s + ":" + x.id;
    if (x.rsfx && !(k in marks)) { marks[k] = 1; n++; }
  }
  if (n) save();
})();
// edits already applied to the manifests come back as the label itself: keep them as reviewed
(() => {
  const norm = t => t.replace(/\\n/g, "\n").split("\n").map(l => l.trim().replace(/\s+/g, " "))
                     .filter(l => l).join("\n");
  let n = 0;
  for (const [s, rows] of Object.entries(DATA)) for (const x of rows) {
    const v = store[s + ":" + x.id];
    if (v && v.status === "edit" && norm(v.text) === x.text) { store[s + ":" + x.id] = {status: "ok", text: null}; n++; }
  }
  if (n) save();
})();
const srcSel = $("src"), minIn = $("minrun"), sortSel = $("sort"), hasSel = $("has");
for (const s of Object.keys(DATA)) srcSel.add(new Option(`${s} (${DATA[s].length})`, s));
let page = 0;
minIn.value = MIN_DEFAULT[srcSel.value];
sortSel.value = SORT_DEFAULT[srcSel.value];
// reviewed last: rows with a review state go behind the rest. The order is taken when the
// page loads or a control changes, then held while paging, so reviewing a page does not
// shift unreviewed rows past the next one
const doneLast = $("donelast");
try { doneLast.checked = localStorage.getItem(KEY + ":donelast") === "1"; } catch (e) {}
let order = null;
const resort = () => { order = null; page = 0; render(); };
srcSel.onchange = () => { minIn.value = MIN_DEFAULT[srcSel.value];
  sortSel.value = SORT_DEFAULT[srcSel.value]; resort(); };
minIn.onchange = sortSel.onchange = hasSel.onchange = resort;
doneLast.onchange = () => {
  try { localStorage.setItem(KEY + ":donelast", doneLast.checked ? "1" : "0"); } catch (e) {}
  resort(); };

function esc(s) { return s.replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }
function marked(t, n) {
  return esc(t).replace(/[가-힣]+/g, m => (n > 0 && m.length >= n) ? `<mark>${m}</mark>` : m);
}
function rows() {
  if (order) return order;
  const s = srcSel.value, n = +minIn.value || 0;
  let r = DATA[s].filter(x => x.run >= n);
  const k = sortSel.value;
  if (k === "sfx") r = r.filter(x => marks[s + ":" + x.id]);
  else if (k === "rsfx") r = r.filter(x => x.rsfx).sort((a, b) => a.conf - b.conf);
  else if (k === "lp") r = r.slice().sort((a, b) => (a.lp ?? 0) - (b.lp ?? 0) || a.id.localeCompare(b.id));
  else if (k === "sus") r = r.slice().sort((a, b) => b.sus - a.sus || a.id.localeCompare(b.id));
  else if (k === "edge") r = r.filter(x => x.edge).sort((a, b) => b.edge.length - a.edge.length);
  else if (k === "dis") r = r.filter(x => x.rd != null && !x.agree && x.rd.replace(/\s/g, "") !== x.text.replace(/\s/g, ""))
                             .sort((a, b) => a.conf - b.conf);
  else if (k === "nolf") r = r.slice().sort((a, b) => (a.lf - b.lf) || (b.len - a.len));
  else if (k === "run") r = r.slice().sort((a, b) => b.run - a.run || a.id.localeCompare(b.id));
  else if (k === "id") r = r.slice().sort((a, b) => a.id.localeCompare(b.id));
  else r = r.slice().sort((a, b) => !!store[s + ":" + a.id] - !!store[s + ":" + b.id]);
  // has: what the row's boxes hold now (a drop holds nothing)
  const hv = hasSel.value;
  if (hv === "part") r = r.filter(x => x.part);
  else if (hv) r = r.filter(x => { const v = store[s + ":" + x.id];
    const m = !!(v && v.status === "drop" ? "" : (v && v.text != null ? v.text : x.text)).trim();
    const b = !!subOf(s + ":" + x.id, x).trim();
    return hv === "main" ? m && !b : hv === "sub" ? !m && b : m && b; });
  if (doneLast.checked) r = r.filter(x => !store[s + ":" + x.id]).concat(r.filter(x => store[s + ":" + x.id]));
  return (order = r);
}
function stats() {
  const s = srcSel.value; let ok = 0, ed = 0, dr = 0, sx = 0;
  for (const x of DATA[s]) { if (marks[s + ":" + x.id]) sx++; const v = store[s + ":" + x.id]; if (!v) continue;
    if (v.status === "ok") ok++; else if (v.status === "drop") dr++; else ed++; }
  $("stats").textContent = `${rows().length} listed · ok ${ok} · edited ${ed} · drop ${dr} · sfx ${sx}`;
}
function render() {
  const s = srcSel.value, r = rows(), pages = Math.max(1, Math.ceil(r.length / PER));
  page = Math.min(page, pages - 1);
  const list = $("list"); list.innerHTML = "";
  for (const x of r.slice(page * PER, (page + 1) * PER)) {
    const key = s + ":" + x.id, v = store[key];
    const d = document.createElement("div");
    if (x.ro) {  // reference row: crop, label, page link only
      d.className = "row";
      d.innerHTML = `<div class="crop"><img loading="lazy" src="${x.img}"></div>
        <div><div class="meta">${esc(x.id)}${x.page ? ` · <a href="${x.page}" target="_blank">page</a>` : ""}
          box ${x.box.join(",")}</div><div class="orig" style="font-size:17px">${esc(x.text)}</div></div>`;
      list.appendChild(d); continue;
    }
    d.className = "row" + (v ? " " + v.status : "");
    d.innerHTML = `<div class="crop"><img loading="lazy" src="${x.img}"></div>
      <div><div class="meta">${esc(x.id)}${x.status ? " · " + x.status : ""} · run ${x.run}
        · sus ${x.sus.toFixed(1)}${x.why ? " (" + esc(x.why) + ")" : ""}${x.lp != null ? " · logprob " + x.lp.toFixed(3) : ""}
        ${x.page ? `· <a href="${x.page}" target="_blank">page</a>` : ""} box ${x.box.join(",")}</div>
      <div class="orig">${marked(x.text, +minIn.value || 0)}</div>
      ${x.note ? `<div class="meta">${esc(x.note)}</div>` : ""}
      ${x.rd != null ? `<div class="meta">__READER__${x.rsfx ? " SFX:" : ""} (${x.conf.toFixed(2)})${x.rwhy ? " [" + x.rwhy + "]" : ""}: ${esc(x.rd || "—").replace(/\n/g, " ⏎ ")}${x.edge ? " · edge " + esc(x.edge) : ""}</div>` : ""}
      ${x.twin_img ? `<div class="twin"><img loading="lazy" src="${x.twin_img}"><div class="meta">twin (${esc(x.twin_lang)}) v4: ${esc(x.twin || "—").replace(/\n/g, " ⏎ ")}</div></div>` : ""}
      ${x.ms ? `<div class="meta">main (OCR:) · the largest text</div>` : ""}
      <textarea class="mainta" rows="${Math.max(2, x.text.split("\n").length + 1)}"></textarea>
      <div class="subbox" hidden><div class="meta">sub (OCR2:) · same crop · one item per line, reading order${x.sub ? " · proposal: " + esc(x.sub).replace(/\n/g, " ⏎ ") : ""}</div>
        <textarea class="subta" rows="2"></textarea></div>
      <div class="btns"><button class="okb">✓ ok</button><button class="drb">✗ drop row</button>
        <button class="rsb">reset</button><button class="subb">+ sub</button>
        <label><input type="checkbox" class="sfxb"> SFX</label></div></div>`;
    const ta = d.querySelector(".mainta");
    const subbox = d.querySelector(".subbox"), subta = d.querySelector(".subta"), subb = d.querySelector(".subb");
    const subPaint = () => { const t = subOf(key, x); subbox.hidden = !t && !(key in subs);
      subb.textContent = subbox.hidden ? "+ sub" : "− sub"; d.classList.toggle("hassub", !!t); };
    subta.value = subOf(key, x);
    subb.onclick = () => {
      if (subbox.hidden) { subs[key] = subOf(key, x); subbox.hidden = false; subb.textContent = "− sub"; subta.focus(); }
      else { subs[key] = ""; subta.value = ""; subPaint(); }
      save(); };
    subta.oninput = () => { subs[key] = subta.value; save(); d.classList.toggle("hassub", !!subta.value); };
    subPaint();
    ta.value = v && v.text != null ? v.text : x.text;
    const okb = d.querySelector(".okb"), drb = d.querySelector(".drb");
    const sfxb = d.querySelector(".sfxb");
    sfxb.checked = !!marks[key];
    sfxb.onchange = () => { if (sfxb.checked) marks[key] = 1; else if (x.rsfx) marks[key] = 0;
      else delete marks[key]; save(); paint(); };
    const paint = () => { const w = store[key];
      d.className = "row" + (w ? " " + w.status : "") + (marks[key] ? " sfx" : "")
        + (subOf(key, x) ? " hassub" : "");
      okb.classList.toggle("on", !!w && w.status === "ok"); okb.classList.toggle("ok", true);
      drb.classList.toggle("on", !!w && w.status === "drop"); drb.classList.toggle("drop", true); stats(); };
    ta.oninput = () => { store[key] = ta.value === x.text ? {status: "ok", text: null}
                                     : {status: "edit", text: ta.value}; save(); paint(); };
    // ok = this row is reviewed with what the box holds now (edited or not)
    okb.onclick = () => { store[key] = ta.value === x.text ? {status: "ok", text: null}
                                     : {status: "edit", text: ta.value}; save(); paint(); };
    drb.onclick = () => { store[key] = {status: "drop", text: null}; save(); paint(); };
    d.querySelector(".rsb").onclick = () => { delete store[key]; delete subs[key]; ta.value = x.text;
      subta.value = subOf(key, x); subPaint(); save(); paint(); };
    paint();
    list.appendChild(d);
  }
  const btn = (t, p) => { const b = document.createElement("button"); b.textContent = t;
    b.disabled = p < 0 || p >= pages; b.onclick = () => { page = p; render(); scrollTo(0, 0); }; return b; };
  for (const id of ["pagerTop", "pager"]) {
    const pg = $(id); pg.innerHTML = "";
    pg.append(btn("← prev", page - 1), Object.assign(document.createElement("span"),
      {textContent: `${page + 1} / ${pages}`}), btn("next →", page + 1));
  }
  stats();
}
$("export").onclick = () => {
  const lines = ["source\tid\tstatus\torig\tnew\tsfx\tsub"];
  for (const k of new Set([...Object.keys(store), ...Object.keys(marks), ...Object.keys(subs)])) {
    if (!store[k] && !marks[k] && !(k in subs)) continue;  // an unchecked SFX: read, nothing else
    const v = store[k] || {status: "", text: null};
    const i = k.indexOf(":"), s = k.slice(0, i), id = k.slice(i + 1);
    const x = (DATA[s] || []).find(y => y.id === id); if (!x) continue;
    // sub column: the row's sub when the reviewer touched it ("-" = removed); a reviewed
    // row whose sub box shows the proposal untouched exports the proposal; else empty
    const sb = k in subs ? (subs[k].trim() ? subs[k].replace(/\\n/g, "\n") : "-")
             : (store[k] && store[k].status !== "drop" && x.sub ? x.sub : "");
    const q = t => (t == null ? "" : t).replace(/\\/g, "\\\\").replace(/\t/g, "\\t").replace(/\n/g, "\\n");
    // a typed "\\n" counts as a line break, same as Enter
    const nt = v.status === "edit" ? v.text.replace(/\\n/g, "\n") : "";
    lines.push([s, id, v.status, q(x.text), q(nt), marks[k] ? "1" : "", q(sb)].join("\t"));
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lines.join("\n") + "\n"], {type: "text/tab-separated-values"}));
  a.download = "ko_label_review.tsv"; a.click();
};
render();
</script></body></html>
"""


def longest_run(t: str) -> int:
    return max((len(m) for m in RUN.findall(t)), default=0)


def suspicion(t: str, w: int, h: int, lang: str = "ko") -> tuple[float, str]:
    """Heuristic doubt in a label, with the reasons.

    * line count: a label of L lines with m glyphs on its longest line (ASCII half) fills
      the crop at a glyph size of w / m (horizontal) or h / m (vertical), giving ~h·m / 1.15w
      or ~w·m / 1.15h lines; the closer layout counts, log2 of its miss when ≥ 1.8× off;
    * KO: kana +3, hanja +1.5 (JA leaking into a KO label); JA: Hangul +3;
    * KO: a spaceless Hangul run of 7+: +0.5 per syllable past 6.
    """
    import math

    lines = [ln for ln in t.split("\n") if ln.strip()] or [""]
    L = len(lines)
    m = (
        max(
            sum(0.5 if c.isascii() else 1.0 for c in ln if not c.isspace())
            for ln in lines
        )
        or 1
    )
    est = min(
        (h * m / (w * 1.15), w * m / (h * 1.15)), key=lambda e: abs(math.log2(e / L))
    )
    s, why = 0.0, []
    r = est / L
    if r >= 1.8 or r <= 1 / 1.8:
        s += abs(math.log2(r))
        why.append(
            f"lines {L} vs ~{est:.0f}" if est >= 1 else f"lines {L} vs ~{est:.1f}"
        )
    if lang == "ja":
        if HANGUL.search(t):
            s += 3
            why.append("hangul")
        return round(s, 2), " · ".join(why)
    if KANA.search(t):
        s += 3
        why.append("kana")
    if HANJA.search(t):
        s += 1.5
        why.append("hanja")
    run = longest_run(t)
    if run >= 7:
        s += (run - 6) * 0.5
        why.append(f"run {run}")
    return round(s, 2), " · ".join(why)


def read_fields(f) -> dict:
    """The flags row's read, confidence and edge text (``rd = None``: no read), whether
    the read came from the ``SFX:`` pass (``rsfx``; the page pre-checks the row's SFX box)
    and ``ko_read.py``'s guard note (``rwhy``: ``length`` / ``runaway`` / ``empty``)."""
    if f is None:
        return dict(rd=None, conf=0.0, edge="", rsfx=False, rwhy="")
    return dict(
        rd=f.read,
        conf=round(float(f.conf), 3) if f.conf == f.conf else 0.0,
        edge=f.edge if f.edge else "",
        rsfx=getattr(f, "task", "") == "SFX:",
        rwhy=getattr(f, "why", "") or "",
    )


def hito_rows(d: Path, flags: dict) -> dict[str, list[dict]]:
    """``manifest_pseudo_hito_v4`` as two sources (``hito_v4_ja`` / ``hito_v4_ko``), each
    row with its twin crop, the twin's v4 read and the read's logprob, and the flags
    reader's read when ``flags`` holds the row."""
    path = d / "manifest_pseudo_hito_v4.parquet"
    if not path.exists():
        return {}
    reads = {}
    for line in open(HITO / "prelabel_v4lp.jsonl", encoding="utf-8"):
        x = json.loads(line)
        reads[(x["id"], x["side"])] = x
    regions = {}
    for f in HITO.glob("*/regions.jsonl"):
        for line in open(f, encoding="utf-8"):
            x = json.loads(line)
            regions[x["id"]] = x
    out = {"hito_v4_ja": [], "hito_v4_ko": []}
    for r in pd.read_parquet(path).itertuples():
        rid, side = r.id.rsplit("_", 1)
        g = regions[rid]
        # one side only when the other edition has no ink there (1,595 regions)
        if side == "ja":
            twin_side, twin_img, cut = g["twin_lang"], g.get("crop_twin"), g["cut_ja"]
        else:
            twin_side, twin_img, cut = "ja", g.get("crop_ja"), g["cut_twin"]
        lp = reads.get((rid, side), {}).get("logprob")
        sus, why = suspicion(r.text, int(r.w), int(r.h), "ja" if side == "ja" else "ko")
        viz = HITO / rid.split("_")[0] / "viz" / (Path(g["ja_page"]).stem + ".jpg")
        out[f"hito_v4_{side}"].append(
            dict(
                id=r.id,
                text=r.text,
                run=longest_run(r.text),
                sus=sus,
                why=why,
                lf=int("\n" in r.text),
                len=len(r.text.replace("\n", "").replace(" ", "")),
                status="",
                **read_fields(flags.get((f"hito_v4_{side}", r.id))),
                lp=lp,
                twin=reads.get((rid, twin_side), {}).get("read"),
                twin_lang=twin_side,
                twin_img=(HITO / twin_img).as_uri() if twin_img else "",
                img=Path(r.path).as_uri(),
                page=viz.as_uri() if viz.exists() else "",  # viz/ covers ~8 % of pages
                box=[int(v) for v in cut],
            )
        )
    for s, rows in out.items():
        print(f"[review] {s}: {len(rows)} rows", flush=True)
    return out


def odd_dots(t: str) -> str:
    """Why a label's dots look off after :func:`ellipsis_fold`, or ``""``: a run of
    two or more dot glyphs it left (not a multiple of three), dots next to ``…``, or a
    run split by a line break."""
    why = []
    if DOTS.search(t):
        why.append("unfolded")
    if MIXED.search(t):
        why.append("mixed")
    if SPLIT.search(t):
        why.append("split")
    return "/".join(why)


def agent_subs(root: Path | None) -> dict[str, tuple[str, str, str]]:
    """``furigana_find.py batches``' agent outputs under ``root`` (``**/out*.tsv``: n, id,
    sub with ``\\n`` breaks, verdict, note) → id → (sub, verdict, note)."""
    out: dict[str, tuple[str, str, str]] = {}
    if root is None:
        return out
    for f in sorted(root.glob("**/out*.tsv")):
        df = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False, quoting=3)
        df = df.reindex(columns=["n", "id", "sub", "verdict", "note"]).fillna("")
        for r in df.itertuples():
            sub = r.sub.replace("\\n", "\n").strip()
            out[r.id] = (sub, r.verdict.strip(), r.note.strip())
    return out


def furigana_rows(
    d: Path, path: Path | None, fix: Path | None = None
) -> dict[str, list[dict]]:
    """``furigana_find.py read``'s crops with ruby groups: the label as it is, the sub box
    prefilled with every group's non-empty read in reading order: one line per main
    column (right to left), a column's groups top down, joined by a space. ``sus`` counts
    the groups that read as something else or sit over no kanji (those rows first)."""
    if path is None or not path.exists():
        return {}
    fixed = agent_subs(fix)
    rows = []
    for line in open(path, encoding="utf-8"):
        x = json.loads(line)
        gs = sorted(x["groups"], key=lambda g: (g["col"], g["y0"]))
        if not gs:
            continue
        good = [g for g in gs if g.get("kana") and g["aligned"]]
        bad = len(gs) - len(good)
        page = m109.page_path(x["book"], int(x["page"]))
        note = " · ".join(f"{g['span'] or '?'}→{g.get('read', '') or '∅'}" for g in gs)
        rows.append(
            dict(
                id=f"{x['book']}_{int(x['page']):03d}_{x['id']}",
                text=x["text"],
                run=0,
                sus=float(bad),
                why=f"{len(gs)} ruby groups, {bad} doubtful",
                lf=int("\n" in x["text"]),
                len=len(x["text"]),
                status="",
                note=f"ruby: {note}",
                agent=f"{x['book']}_{int(x['page']):03d}_{x['id']}" in fixed,
                sub="\n".join(
                    " ".join(g["read"] for g in gs if g["col"] == c and g.get("read"))
                    for c in sorted({g["col"] for g in gs if g.get("read")})
                ),
                img=(d / x["path"]).as_uri(),
                page=page.as_uri() if page.exists() else "",
                box=[],
            )
        )
    for r in rows:
        if r.pop("agent"):
            sub, verdict, anote = fixed[r["id"]]
            r["sub"] = sub
            r["note"] = (
                f"agent {verdict}" + (f": {anote}" if anote else "") + " · " + r["note"]
            )
    print(
        f"[review] furigana: {len(rows)} rows ({len(fixed)} agent-reviewed)", flush=True
    )
    return {"furigana": rows}


def linebreak_rows(d: Path) -> dict[str, list[dict]]:
    """``linebreak_auto.py apply``'s rows that missed the gate: the proposed breaks as
    the label, the column reads as a note, sorted by read distance, the rows with a dot
    run :func:`ellipsis_fold` left alone (:func:`odd_dots`) first."""
    path = REPO / "output/ocr/linebreak/linebreak_suspicious.jsonl"
    if not path.exists():
        return {}
    df = pd.read_parquet(d / "manifest_sfxp_speech.parquet")
    df = df[df.source == "sfxp_speech"]
    crop = {(r.book, int(r.page), str(r.id)): r.path for r in df.itertuples()}
    rows = []
    for line in open(path, encoding="utf-8"):
        x = json.loads(line)
        k = (x["book"], int(x["page"]), str(x["id"]))
        # dot-only labels are linebreak_auto.py dots' (dots_suspicious), not this source's
        if k not in crop or not DOT_FREE.sub("", x["raw"]):
            continue
        text = round_dots(ellipsis_fold(x["out"]))
        odd = odd_dots(text)
        page = m109.page_path(x["book"], int(x["page"]))
        reads = " ⏎ ".join(x["lines"])
        rows.append(
            dict(
                id=f"{x['book']}_{int(x['page']):03d}_{x['id']}",
                text=text,
                run=0,
                sus=x["dist"] + (100.0 if odd else 0.0),
                why=f"{x['n_segs']} segments" + (f" · dots {odd}" if odd else ""),
                lf=int("\n" in x["out"]),
                len=len(x["raw"]),
                status="",
                note=f"distance {x['dist']:.2f} · label: {x['raw']} · column reads: {reads}",
                img=(d / crop[k]).as_uri(),
                page=page.as_uri() if page.exists() else "",
                box=[],
            )
        )
    print(f"[review] linebreak_suspicious: {len(rows)} rows", flush=True)
    return {"linebreak_suspicious": rows}


LINE_DOT_RUN = re.compile(r"[.．・･…‥]+")
RAW_DOT = re.compile(r"[.．・･]{2,}|[.．・･][…‥\n]|[…‥\n][.．・･]")


def round_dots(t: str) -> str:
    """Each dot run inside a line (``…`` counts 3, ``‥`` 2) → ``max(1, round(n / 3))`` ×
    ``…``, as the review edits of 11-18 spell them (2 → ``…``, 7 → ``……``); a lone
    ``・`` (nakaguro) and a run a line break splits stay as written."""

    def sub(m: re.Match) -> str:
        r, a, b = m[0], m.start(), m.end()
        dot = ".．・･…‥"
        if (a >= 2 and t[a - 1] == "\n" and t[a - 2] in dot) or (
            b + 1 < len(t) and t[b] == "\n" and t[b + 1] in dot
        ):
            return r
        n = sum(3 if c == "…" else 2 if c == "‥" else 1 for c in r)
        return r if n < 2 else "…" * max(1, round(n / 3))

    return LINE_DOT_RUN.sub(sub, t)


def odd_dot_rows(d: Path) -> dict[str, list[dict]]:
    """``manifest_sfxp_speech`` rows with text whose dots :func:`ellipsis_fold` left
    odd (:func:`odd_dots`; ``linebreak_suspicious``'s rows are listed there): the
    :func:`round_dots` spelling as the label (``ok`` takes it), the label as a note.
    Suspicion order: a run a line break splits first (its place is the page's), then
    dots next to ``…``, then the other runs, longest run first."""
    sus_path = REPO / "output/ocr/linebreak/linebreak_suspicious.jsonl"
    skip = set()
    if sus_path.exists():
        for line in open(sus_path, encoding="utf-8"):
            x = json.loads(line)
            skip.add((x["book"], int(x["page"]), str(x["id"])))
    df = pd.read_parquet(d / "manifest_sfxp_speech.parquet")
    df = df[df.source.isin(["sfxp_speech", "m109_lines"])]
    rows = []
    for r in df.itertuples():
        k = (r.book, int(r.page), str(r.id))
        if k in skip or not DOT_FREE.sub("", r.text):
            continue
        odd = odd_dots(r.text)
        # ``…\n…`` alone (dot columns already spelled) is not odd here
        if not odd or not RAW_DOT.search(r.text):
            continue
        longest = max(len(m) for m in LINE_DOT_RUN.findall(r.text))
        page = m109.page_path(r.book, int(r.page))
        rows.append(
            dict(
                id=f"{r.book}_{int(r.page):03d}_{r.id}",
                text=round_dots(r.text),
                run=0,
                sus=100.0 * ("split" in odd)
                + 10.0 * ("mixed" in odd)
                + 1.0
                + longest / 100,
                why=f"dots {odd}",
                lf=int("\n" in r.text),
                len=len(r.text),
                status="",
                note=f"label: {r.text}",
                img=(d / r.path).as_uri(),
                page=page.as_uri() if page.exists() else "",
                box=[],
            )
        )
    print(f"[review] odd_dots: {len(rows)} rows", flush=True)
    return {"odd_dots": rows}


def doujin_rows(reads_path: Path | None) -> dict[str, list[dict]]:
    """The doujin gate's SFX rows (id = the TSV's ``row``, zero-padded so the id sort is
    the file order), with ``eval_sfx.py``'s per-row read when ``reads_path`` is given."""
    df = pd.read_csv(DOUJIN_LABELS, sep="\t", dtype=str, keep_default_na=False)
    df = df[df.kind_hand == "sfx"]
    reads = {}
    if reads_path:
        for line in open(reads_path, encoding="utf-8"):
            x = json.loads(line)
            reads[x["row"]] = x
    rows = []
    for r in df.itertuples():
        box = json.loads(r.box)
        sus, why = suspicion(r.text_hand, box[2] - box[0], box[3] - box[1], "ja")
        x = reads.get(r.row)
        page = DOUJIN_FROZEN / "pages" / f"{r.stem}.png"
        note = " · ".join(
            t for t in (f"pre-label: {r.text_rec}" if r.text_rec else "", r.note) if t
        )
        rows.append(
            dict(
                id=f"{int(r.row):04d}",
                text=r.text_hand,
                run=0,
                sus=sus,
                why=why,
                lf=int("\n" in r.text_hand),
                len=len(r.text_hand),
                status=r.status,
                note=note,
                rd=x["pred"] if x else None,
                conf=round(float(x["sim"]), 3) if x else 0.0,
                edge="",
                rsfx=False,
                rwhy="runaway" if x and x.get("runaway") else "",
                agree=bool(x and x["exact"]),  # eval_sfx's exact (its folds): off "dis"
                img=(DOUJIN_FROZEN / "crops" / f"{r.row}.png").as_uri(),
                page=page.as_uri() if page.exists() else "",
                box=[int(v) for v in box],
            )
        )
    print(f"[review] doujin_sfx: {len(rows)} rows ({len(reads)} reads)", flush=True)
    return {"doujin_sfx": rows}


def mainsub_rows(
    d: Path, path: Path | None, main_n: int, seed: int = 0
) -> dict[str, list[dict]]:
    """``m109_mainsub.py``'s crops, one source each: every row with a proposed sub or
    partly visible text, plus ``main_n`` seeded main-only rows. ``sus`` puts partly
    visible text and a main that moved off the crop's label first."""
    if path is None or not path.exists():
        return {}
    by: dict[str, list[dict]] = {}
    for line in open(path, encoding="utf-8"):
        x = json.loads(line)
        by.setdefault(x["source"], []).append(x)
    rng = np.random.default_rng(seed)
    out = {}
    for src, xs in by.items():
        cand = [x for x in xs if x["sub"] or x["part"]]
        rest = [x for x in xs if not (x["sub"] or x["part"])]
        pick = rng.permutation(len(rest))[: min(main_n, len(rest))]
        rows = []
        for x in cand + [rest[i] for i in sorted(pick)]:
            moved = x["main"] != x["label"]
            items = " | ".join(
                f"{i['k']} {i['t']} (g {i['g']:.0f})".replace("\n", " ⏎ ")
                for i in x["items"]
            )
            part = " | ".join(
                f"{p['k']} {p['t']} ({p['f']:.0%})".replace("\n", " ⏎ ")
                for p in x["part"]
            )
            note = f"in crop: {items}"
            if part:
                note += f" · partly visible: {part}"
            if moved:
                note += f" · label now: {x['label']}".replace("\n", " ⏎ ")
            page = m109.page_path(x["book"], int(x["page"]))
            rows.append(
                dict(
                    id=x["id"],
                    text=x["main"],
                    run=0,
                    sus=2.0 * len(x["part"]) + 1.0 * moved + 0.5 * bool(x["sub"]),
                    why=f"{len(x['items'])} in crop, {len(x['part'])} partly",
                    lf=int("\n" in x["main"]),
                    len=len(x["main"]),
                    status=f"{x['split']} · {x['cat']}",
                    note=note,
                    sub=x["sub"],
                    part=len(x["part"]),
                    ms=1,
                    img=(d / x["path"]).as_uri(),
                    page=page.as_uri() if page.exists() else "",
                    box=x["box"],
                )
            )
        out[src] = rows
        print(
            f"[review] {src}: {len(rows)} rows ({len(cand)} with sub / partly "
            f"visible text, {len(pick)} main-only of {len(rest)})",
            flush=True,
        )
    return out


def ref_rows(d: Path, n: int, seed: int = 0) -> dict[str, list[dict]]:
    """Read-only Manga109 references: ``n`` seeded rows each of COO onomatopoeia
    (``kind = sfx``) and speech text from ``manifest.parquet``."""
    df = pd.read_parquet(d / "manifest.parquet")
    out = {}
    for kind in ("sfx", "speech"):
        sub = df[df.kind == kind]
        sub = sub.sample(min(n, len(sub)), random_state=seed).sort_values(
            ["book", "page"]
        )
        rows = []
        for r in sub.itertuples():
            xy = json.loads(r.poly) if isinstance(r.poly, str) else list(r.poly)
            xs, ys = xy[0::2], xy[1::2]
            page = m109.page_path(r.book, int(r.page))
            rows.append(
                dict(
                    id=f"{r.book}_{int(r.page):03d}_{r.id}",
                    text=r.text,
                    run=0,
                    sus=0.0,
                    why="",
                    lf=int("\n" in r.text),
                    len=len(r.text),
                    ro=True,
                    img=(d / r.path).as_uri(),
                    page=page.as_uri() if page.exists() else "",
                    box=[int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                )
            )
        out[f"m109_{kind}"] = rows
        print(f"[review] m109_{kind}: {len(rows)} rows (of {len(df[df.kind == kind])})")
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--min_run", type=int, default=0, help="the page's default run filter"
    )
    ap.add_argument(
        "--flags",
        type=Path,
        help="ko_label_fix.py's <reads>_flags.parquet: a reader's read, confidence, edge "
        "text per row (the reader's name from its `reader` column)",
    )
    ap.add_argument(
        "--ref_n",
        type=int,
        default=3000,
        help="Manga109 reference rows per kind (m109_sfx / m109_speech); 0 = none",
    )
    ap.add_argument(
        "--doujin_reads",
        type=Path,
        help="eval_sfx.py's output/ocr/eval/sfx_<name>.jsonl: a reader's read per "
        "doujin_sfx row",
    )
    ap.add_argument(
        "--ruby",
        type=Path,
        help="furigana_find.py read's output/ocr/furigana/<name>_read.jsonl: the "
        "furigana source (sub box prefilled with the ruby reads)",
    )
    ap.add_argument(
        "--ruby_fix",
        type=Path,
        help="folder of agent outputs (**/out.tsv from furigana_find.py batches): "
        "their sub replaces the furigana prefill, the verdict leads the meta line",
    )
    ap.add_argument(
        "--mainsub",
        type=Path,
        help="m109_mainsub.py's output/ocr/mainsub/m109_mainsub.jsonl: the Manga109 "
        "main / sub sources (mainsub_sfx_wide, mainsub_det, mainsub_speech)",
    )
    ap.add_argument(
        "--mainsub_main_n",
        type=int,
        default=2000,
        help="seeded main-only rows per --mainsub source (rows with a sub or partly "
        "visible text are all listed)",
    )
    a = ap.parse_args()
    d = m109.derived_root()
    kz = pd.read_parquet(d / "manifest_sfxp_kozh.parquet")
    gel = {}
    for line in open(GELNOTE / "manifest.jsonl", encoding="utf-8"):
        x = json.loads(line)
        gel[x["id"]] = x
    lab = {}
    for line in open(SWAPRELAX / "labels.jsonl", encoding="utf-8"):
        x = json.loads(line)
        lab[x["crop"]] = x
    sw = (
        pd.read_parquet(d / "manifest_pseudo_swaprelax_v5.parquet").set_index("id").path
    )
    flags, reader = {}, "reader"
    if a.flags:
        fl = pd.read_parquet(a.flags)
        # flags from before ko_read.py named the reader are v6's
        reader = str(fl.reader.iloc[0]) if "reader" in fl else "v6"
        for f in fl.itertuples():
            flags[(f.source, f.id)] = f
    data = {}
    for src in ("gelnote_ko", "swaprelax_v5"):
        out = []
        for r in kz[kz.source == src].itertuples():
            run = longest_run(r.text)
            status = ""
            if src == "gelnote_ko":
                x = gel[r.id]
                page, box = GELNOTE / x["page_path"], x["box"]
            else:
                x = lab["crops/" + os.path.basename(sw[r.id])]
                page, box = REPO / x["page"], x["box"]
                status = x["status"]
            sus, why = suspicion(r.text, int(r.w), int(r.h))
            f = flags.get((src, r.id))
            out.append(
                dict(
                    id=r.id,
                    text=r.text,
                    run=run,
                    sus=sus,
                    why=why,
                    lf=int("\n" in r.text),
                    len=len(r.text.replace("\n", "").replace(" ", "")),
                    status=status,
                    **read_fields(f),
                    img=(d / r.path).as_uri(),
                    page=Path(page).as_uri(),
                    box=[int(v) for v in box],
                )
            )
        data[src] = out
        print(f"[review] {src}: {len(out)} rows", flush=True)
    data.update(hito_rows(d, flags))
    data.update(odd_dot_rows(d))
    data.update(linebreak_rows(d))
    data.update(furigana_rows(d, a.ruby, a.ruby_fix))
    data.update(doujin_rows(a.doujin_reads))
    data.update(mainsub_rows(d, a.mainsub, a.mainsub_main_n))
    if a.ref_n:
        data.update(ref_rows(d, a.ref_n))
    a.out.mkdir(parents=True, exist_ok=True)
    html = (
        PAGE.replace("__READER__", reader)
        .replace("__DOUJIN_SORT__", "dis" if a.doujin_reads else "sus")
        .replace("__DATA__", json.dumps(data, ensure_ascii=False))
        .replace("__MINRUN__", json.dumps({s: a.min_run for s in data}))
    )
    (a.out / "index.html").write_text(html, encoding="utf-8")
    print(f"[review] → {a.out / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
