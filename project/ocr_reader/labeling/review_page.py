#!/usr/bin/env python3
"""A local review page for the Korean ``OCR:`` labels of the two-prompt reader
(``manifest_sfxp_kozh.parquet``: gelnote KO and swaprelax_v5) and the hito v4 self-labels
(``manifest_pseudo_hito_v4.parquet``, one source per side: ``hito_v4_ja`` / ``hito_v4_ko``):
the crop as trained, the label in an editable box, a status per row. Edits stay in the
browser (localStorage) and leave as a TSV through the page's Export button
(``apply_review.py`` takes it). The export carries only the rows whose state differs
from what apply has written: the manifests' text (as the page was built), the gate
TSV's status, and ``assets/{sfx_marks,sub_labels,m109_mainsub}.tsv`` (``applied_state``);
rebuild the page after each apply.

Every editable row has an ``SFX`` checkbox (exported as the ``sfx`` column;
``apply_review.py`` keeps the marks in ``assets/sfx_marks.tsv``, training does not read
them); a row the ``--flags`` reader read under ``SFX:`` comes up checked. ``--ref_n N``
adds ``m109_sfx`` / ``m109_speech``, read-only references: a seeded sample of the Manga109
COO onomatopoeia and the speech text (``manifest.parquet``); off by default.

Every editable row has a ``+ sub`` button: a second box for the ``OCR2:`` label of the same
crop (furigana, a smaller SFX, a hand-lettered aside; one item per line, reading order),
exported as the ``sub`` column (``-`` = removed). The furigana reads (``--ruby``,
``furigana_find.py``; ``--ruby_fix`` the agents' fixes) are not a source of their own
(dropped 11-21): they head the ``mainsub_*`` sub proposals, and the ``furigana`` /
``furigana suspicious`` sorts list the mainsub rows with a ruby proposal (by id / most
doubtful groups first).

``mainsub_det`` / ``mainsub_speech`` (``--mainsub``, ``m109_mainsub.py``):
every Manga109 training crop, the box holding the proposed ``OCR:`` main (the largest text),
the sub box the proposed ``OCR2:`` sub (the other annotated text in the crop; on a speech
crop with ruby groups, the furigana source's reads first, and that row's sub edits carry
over). The meta line lists each item's glyph size and the text only partly inside the
crop (type what is visible into the sub). An emptied main box is a crop with only sub text.
Rules: ``README.md`` § Labeling. The header's ``has`` filter (any source) lists rows by what their boxes
hold now: main only, sub only, both, or partly visible text.
``apply_review.py`` writes these rows to ``assets/m109_mainsub.tsv``.

Line breaks that missed ``linebreak_auto.py apply``'s gate are not a source of their own
(dropped 11-21): a ``mainsub_*`` row whose main is one such ``<text>`` box holds the
proposed breaks in its main box (the meta line: the label, the column reads), and the
``linebreak suspicious`` sort lists those rows, largest read distance first. Its ``ok``
goes to ``m109_mainsub.tsv`` (the det crop's main), not the speech manifest.
(``odd_dots`` and ``dots_suspicious`` are no longer listed: the first was empty by 11-21,
the second reviewed in full on 11-18; ``apply_review.py`` still takes their rows.)

hito rows also show the twin crop (the other edition's lettering of the same region) with
its v4 read, and the read's mean token logprob (``prelabel_v4lp.jsonl``); they sort lowest
logprob first by default.

``--hayai`` (``hayai_read.py``'s reads of every detector box, hayai-ocr v2.7 Supernova): a
``mainsub_*`` row shows the read under its meta line, and the ``hayai disagrees`` sort lists
the rows whose main the read does not match (whitespace-blind, with ``exact_key``'s folds; a
read equal to main + sub counts as a match; a main of one kana, or one kana two or three times,
compares with hiragana = katakana and small = large: ``へへへ`` / ``ヘヘヘ``, ``ぁ`` / ``あ``),
largest distance first. hayai reads a crop as
one string with no line breaks, so it says nothing about a row's breaks.
``--hayai_kozh`` (``hayai_read.py --set kozh hito``): the same read and sort on the
``gelnote_ko`` / ``swaprelax_v5`` / ``hito_v4_*`` rows, against the label. hayai often slips
kana or hanja into a Hangul line (and Hangul into a JA one): a read holding a script the
label does not (``hyx``) goes behind the others in the ``hayai disagrees`` sorts. The
``hayai near miss`` sort lists the disagreeing rows smallest distance first. The ``hayai:
likely label errors first`` sort orders them by ``hayai_kind`` (the read is the label plus a
dakuten, the label reordered, or the label plus text the sub lacks, first; the read short of
the label's marks or characters, or the sub alone, last), rows with partly visible text after
the rest of their group, then smallest distance first. The distance takes small kana = large
and a run of dashes / waves, or of dots, as one (``loose_key``).

A row whose main label is over the training cap (``crop_dataset.MAX_TARGET_CHARS``, which
drops it) is not listed or counted (``mark_over_cap``); it stays in the page's data, so
a state it holds still exports. The gates (``doujin_sfx``, ``k3_*``) are not capped. Every other row is
listed; the page filters on the longest spaceless Hangul run (the gelnote
teacher's dropped spacing passes the whitespace-blind agreement, ``pseudo_label.py``) and
sorts by it, by "no line break, longest first", or by ``suspicion`` (``suspicion``: the
label's line count against the crop's shape, kana / hanja, long spaceless runs). With
``--flags`` (``ko_read.py`` → ``ko_label_fix.py``) every kozh and hito row also shows a
reader's read and confidence, and the "disagrees" sort lists the rows whose label the read
does not match (whitespace-blind), lowest confidence first. Re-read with each new run so
that sort follows the latest reader. The ``…`` sort lists the rows whose box holds Hangul
and ``…`` (as edited, so a row respelled to ``...`` leaves it), most ``…`` first.

``doujin_sfx``: the doujin gate's SFX rows (``assets/sfx_labels_sincos_597.tsv``,
``kind_hand == sfx``, ``unchecked`` included) on the frozen 12 % crops
(``eval_data/sincos``); the meta line shows the row's label status, its pre-label
(``text_rec``) and note. ``--doujin_reads`` (``eval_sfx.py``'s
``output/ocr/eval/sfx_<name>.jsonl``) adds that reader's read; its "confidence" is
``eval_sfx``'s sim to the label, and "disagrees" leaves out the rows ``eval_sfx`` scores
exact (its heart / shape folds). ``apply_review.py`` writes these rows back to the TSV.

``k3_ko`` / ``k3_zh``: the KO / ZH gate (``assets/k3_gelnote.tsv``, ``skip`` rows included)
on its frozen crops (``eval_data/k3_gelnote``); the meta line shows the row's status, its
``zh_script``, the label note and the gelbooru translation. ``--k3_reads`` (``eval_k3.py``'s
``output/ocr/eval/k3_<name>.jsonl``) adds that reader's read, its sim as confidence;
"disagrees" leaves out the rows ``eval_k3`` scores exact. ``apply_review.py`` writes these
rows back to the TSV (``ok`` / edit → ``checked``, drop → ``skip``), line breaks kept.

``anima_lowconf`` (``--anima_lowconf``, ``anima_lowconf.py``'s ``rows.jsonl``): the v7 reads
of the Anima training set at or under its score cut, the main box holding the read as a first
pass, the meta line the read's score and det. With ``--anima_agents`` (``label_agent.md``'s
``*/out.tsv``, three agents per batch) the main box holds the agents' majority read where two
agree, the meta line every agent's read and confidence, and the ``agents unsure`` sort lists the
rows the agents were not unanimous and sure on, Hangul first. The ``v7 score`` sort lists them
lowest score first. ``apply_review.py`` writes them to ``assets/anima_mainsub.tsv`` (the
``m109_mainsub.tsv`` columns); a row applied there starts reviewed.

``reader_split`` (``--reader_split``, cjk_rl's ``split_1126_0910``): kana render crops one
reader (hayai v2.7 or the tuned reader) reads as the glyph the caption asked for and the other
does not, the main box holding the read that matched, the meta line both reads. The ``ks``
sort lists hayai-only crops first, grouped by asked glyph. ``apply_review.py`` writes them to
``assets/reader_split_mainsub.tsv``; training does not read it yet.

    python project/ocr_reader/labeling/review_page.py --out ~/ko_label_review
"""

from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from project.ocr_reader.core import manga109 as m109
from project.ocr_reader.core.crop_dataset import (
    MAX_TARGET_CHARS,
    break_proposals,
    filled_breaks,
    normalize_target,
    waiting_break,
    with_breaks,
)
from project.ocr_reader.core.labels import agent_subs, ellipsis_fold
from project.ocr_reader.core.paths import (
    ANIMA_LOWCONF,
    ASSETS,
    EVAL_DATA,
    GELNOTE,
    HITO,
    REPO,
    SWAPRELAX,
)
from project.ocr_reader.core.textnorm import exact_key

DOUJIN_LABELS = ASSETS / "sfx_labels_sincos_597.tsv"
DOUJIN_FROZEN = EVAL_DATA / "sincos"
K3_LABELS = ASSETS / "k3_gelnote.tsv"
K3_FROZEN = EVAL_DATA / "k3_gelnote"
HANGUL = re.compile(r"[가-힣]")
RUN = re.compile(r"[가-힣]+")
KANA = re.compile(r"[\u3040-\u30ff]")
HANJA = re.compile(r"[\u4e00-\u9fff]")
DOTS = re.compile(r"[.．・･]{2,}")
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
.crop img.up { min-height:140px; min-width:140px; }  /* small render crops (reader_split) */
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
    <option value="nolf">no line break, longest first</option>
    <option value="dis">__READER__ disagrees, lowest confidence first</option>
    <option value="run">longest run first</option>
    <option value="ell">한글 + …: most … first</option>
    <option value="lb">linebreak suspicious: proposed breaks, largest read distance first (mainsub)</option>
    <option value="furi">furigana: rows with a ruby proposal (mainsub)</option>
    <option value="furisus">furigana suspicious: most doubtful ruby first (mainsub)</option>
    <option value="hy">hayai disagrees with the main, largest distance first</option>
    <option value="hyn">hayai near miss: disagrees, smallest distance first</option>
    <option value="hyl">hayai: likely label errors first (added mark / reordered / missing text), then near misses</option>
    <option value="sc">v7 score, lowest first (anima_lowconf)</option>
    <option value="ag">agents unsure: Hangul first, then most disagreement (anima_lowconf)</option>
    <option value="ks">hayai only first, by asked glyph, surest read first (reader_split)</option>
    <option value="id">id</option><option value="todo">unreviewed first</option></select></label>
  <label>has <select id="has"><option value="">all</option><option value="main">main only</option>
    <option value="sub">sub only</option><option value="both">main + sub</option>
    <option value="part">partly visible text</option></select></label>
  <label><input type="checkbox" id="donelast"> reviewed last</label>
  <span class="stats" id="stats"></span>
  <span class="stats">edits save as you type · Enter or \n = line break · Tab = ok, next row</span>
  <button id="export">Export TSV</button>
  <span class="pager" id="pagerTop"></span>
</header>
<main><div id="list"></div><div class="pager" id="pager"></div></main>
<script>
const DATA = __DATA__;
const MIN_DEFAULT = __MINRUN__;
const SORT_DEFAULT = {"gelnote_ko": "nolf", "swaprelax_v5": "sus", "hito_v4_ja": "lp", "hito_v4_ko": "lp",
                      "doujin_sfx": "__DOUJIN_SORT__", "k3_ko": "__K3_SORT__", "k3_zh": "__K3_SORT__",
                      "mainsub_det": "sus", "mainsub_speech": "sus", "anima_lowconf": "sc",
                      "reader_split": "ks"};
// mainsub rows carry paths relative to the Manga109 folder
const M109 = __M109__;
const url = (p) => !p || p.includes(":") ? p : M109 + p;
const PER = 50;
const KEY = "ko_label_review_v1";
let store = {};
try { store = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { store = {}; }
let marks = {};
try { marks = JSON.parse(localStorage.getItem(KEY + ":sfx") || "{}"); } catch (e) { marks = {}; }
// sub (OCR2:) text per row: absent = the row's proposal (x.sub) if any, "" = no sub
let subs = {};
try { subs = JSON.parse(localStorage.getItem(KEY + ":sub") || "{}"); } catch (e) { subs = {}; }
// localStorage is written 300 ms after the last change (each write stringifies every
// edit), and at once when the page is hidden or closed
let saveT = null;
const saveNow = () => { clearTimeout(saveT); saveT = null;
  try { localStorage.setItem(KEY, JSON.stringify(store));
  localStorage.setItem(KEY + ":sfx", JSON.stringify(marks));
  localStorage.setItem(KEY + ":sub", JSON.stringify(subs)); } catch (e) {} };
const save = () => { clearTimeout(saveT); saveT = setTimeout(saveNow, 300); };
addEventListener("pagehide", () => { if (saveT) saveNow(); });
document.addEventListener("visibilitychange", () => { if (document.hidden && saveT) saveNow(); });
// a mainsub speech row falls back to its furigana row's sub box (fk) before the proposal
const subOf = (key, x) => key in subs ? subs[key] : (x.fk && x.fk in subs ? subs[x.fk] : (x.sub || ""));
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
// a box's text as apply stores it (textnorm.collapse_keep_lines; a typed "\n" is a break)
const norm = t => (t || "").replace(/\\n/g, "\n").split("\n").map(l => l.trim().replace(/\s+/g, " "))
                   .filter(l => l).join("\n");
// what apply_review.py has written (assets/): export carries only the rows that differ
const APPLIED = __APPLIED__;
const AMARK = new Set(APPLIED.marks);
function changed(k, x) {
  const v = store[k], st = v ? v.status : "", s = k.slice(0, k.indexOf(":"));
  if (!!marks[k] !== AMARK.has(k)) return true;
  if (x.ms) {  // m109_mainsub.tsv holds [status, main, sub] per reviewed crop
    if (!st && !(k in subs)) return false;
    // a state re-derived from the TSV (ap) and left untouched is what apply wrote: an
    // agent row ("agent") stays one until the reviewer touches it
    if (v && v.ap && !(k in subs)) return false;
    const want = st === "drop" ? ["drop", "", ""]
      : ["ok", st === "edit" ? norm(v.text) : x.text, norm(subOf(k, x))];
    const have = APPLIED.ms[k];
    return !have || want.some((w, i) => w !== have[i]);
  }
  let main;
  if (x.gate)  // a gate TSV: ok / edit → checked, drop → x.gate; doujin_sfx stores a break as a space
    main = (st === "edit" && (x.flat ? norm(v.text).replace(/\n/g, " ") : norm(v.text)) !== x.text)
      || ((st === "ok" || st === "edit") && x.status !== "checked")
      || (st === "drop" && x.status !== x.gate);
  else if (x.cur != null)  // proposal sources: ok writes the proposal
    main = (st === "ok" && !x.done) || (st === "edit" && norm(v.text) !== x.cur) || st === "drop";
  else main = (st === "edit" && norm(v.text) !== norm(x.text)) || st === "drop";
  if (main) return true;
  // sub: touched, or the proposal on a reviewed row, against sub_labels.tsv
  const op = k in subs ? subs[k] : (st && st !== "drop" ? subOf(k, x) : null);
  return op != null && norm(op) !== norm(APPLIED.subs[k] || "");
}
// rows m109_mainsub.tsv holds (x.ap) start reviewed as applied; these states are re-derived on
// every load (ap: 1), so a later apply shows through, and a touch of the row replaces them
(() => {
  for (const k of Object.keys(store)) if (store[k] && store[k].ap) delete store[k];
  for (const [s, rows] of Object.entries(DATA)) for (const x of rows) {
    const k = s + ":" + x.id;
    if (x.ap && !store[k]) store[k] = {status: x.ap === "drop" ? "drop" : "ok", text: null, ap: 1};
  }
  save();
})();
// edits already applied to the manifests come back as the label itself: keep them as reviewed
(() => {
  let n = 0;
  for (const [s, rows] of Object.entries(DATA)) for (const x of rows) {
    const v = store[s + ":" + x.id];
    if (v && v.status === "edit" && norm(v.text) === x.text) { store[s + ":" + x.id] = {status: "ok", text: null}; n++; }
  }
  if (n) save();
})();
// over the training cap (the dataset drops them): not listed, still exported
const LISTED = {};
const listed = s => LISTED[s] || (LISTED[s] = DATA[s].filter(x => !x.over));
const srcSel = $("src"), minIn = $("minrun"), sortSel = $("sort"), hasSel = $("has");
for (const s of Object.keys(DATA)) srcSel.add(new Option(`${s} (${listed(s).length})`, s));
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
// what the row's main box holds now (a drop holds nothing)
function boxNow(key, x) {
  const v = store[key];
  return v && v.status === "drop" ? "" : (v && v.text != null ? v.text : x.text);
}
function rows() {
  if (order) return order;
  const s = srcSel.value, n = +minIn.value || 0;
  let r = listed(s).filter(x => x.run >= n);
  const k = sortSel.value;
  if (k === "lp") r = r.slice().sort((a, b) => (a.lp ?? 0) - (b.lp ?? 0) || a.id.localeCompare(b.id));
  else if (k === "sus") r = r.slice().sort((a, b) => b.sus - a.sus || a.id.localeCompare(b.id));
  else if (k === "dis") r = r.filter(x => x.rd != null && !x.agree && x.rd.replace(/\s/g, "") !== x.text.replace(/\s/g, ""))
                             .sort((a, b) => a.conf - b.conf);
  else if (k === "nolf") r = r.slice().sort((a, b) => (a.lf - b.lf) || (b.len - a.len));
  else if (k === "run") r = r.slice().sort((a, b) => b.run - a.run || a.id.localeCompare(b.id));
  else if (k === "ell") r = r.map(x => [x, boxNow(s + ":" + x.id, x)])
    .filter(([, t]) => /[가-힣]/.test(t) && t.includes("…"))
    .map(([x, t]) => [x, t.split("…").length - 1])
    .sort(([a, m], [b, n]) => n - m || a.id.localeCompare(b.id)).map(([x]) => x);
  else if (k === "lb") r = r.filter(x => x.lbs != null).sort((a, b) => b.lbs - a.lbs || a.id.localeCompare(b.id));
  else if (k === "furi") r = r.filter(x => x.rb).sort((a, b) => a.id.localeCompare(b.id));
  else if (k === "furisus") r = r.filter(x => x.rb).sort((a, b) => b.rsus - a.rsus || a.id.localeCompare(b.id));
  else if (k === "hy") r = r.filter(x => x.hyd > 0).sort((a, b) => (a.hyx || 0) - (b.hyx || 0) || b.hyd - a.hyd || a.id.localeCompare(b.id));
  else if (k === "hyn") r = r.filter(x => x.hyd > 0).sort((a, b) => (a.hyx || 0) - (b.hyx || 0) || a.hyd - b.hyd || a.id.localeCompare(b.id));
  else if (k === "hyl") r = r.filter(x => x.hyd > 0).sort((a, b) => (a.hyx || 0) - (b.hyx || 0) || (b.hyk ?? 1) - (a.hyk ?? 1)
    || (a.part > 0) - (b.part > 0) || a.hyd - b.hyd || a.id.localeCompare(b.id));
  else if (k === "ag") r = r.filter(x => x.unc > 0).sort((a, b) => (b.hg || 0) - (a.hg || 0)
    || b.unc - a.unc || (a.sc ?? 1) - (b.sc ?? 1) || a.id.localeCompare(b.id));
  else if (k === "ks") r = r.filter(x => x.ks).sort((a, b) => a.ks[0] - b.ks[0]
    || a.ks[1].localeCompare(b.ks[1]) || a.ks[2] - b.ks[2] || a.id.localeCompare(b.id));
  else if (k === "sc") r = r.slice().sort((a, b) => (a.sc ?? 1) - (b.sc ?? 1) || a.id.localeCompare(b.id));
  else if (k === "id") r = r.slice().sort((a, b) => a.id.localeCompare(b.id));
  else r = r.slice().sort((a, b) => !!store[s + ":" + a.id] - !!store[s + ":" + b.id]);
  // has: what the row's boxes hold now (a drop holds nothing)
  const hv = hasSel.value;
  if (hv === "part") r = r.filter(x => x.part);
  else if (hv) r = r.filter(x => { const m = !!boxNow(s + ":" + x.id, x).trim();
    const b = !!subOf(s + ":" + x.id, x).trim();
    return hv === "main" ? m && !b : hv === "sub" ? !m && b : m && b; });
  if (doneLast.checked) r = r.filter(x => !store[s + ":" + x.id]).concat(r.filter(x => store[s + ":" + x.id]));
  return (order = r);
}
function stats() {
  const s = srcSel.value; let ok = 0, ed = 0, dr = 0, sx = 0;
  for (const x of listed(s)) { if (marks[s + ":" + x.id]) sx++; const v = store[s + ":" + x.id]; if (!v) continue;
    if (v.status === "ok") ok++; else if (v.status === "drop") dr++; else ed++; }
  $("stats").textContent = `${rows().length} listed · ok ${ok} · edited ${ed} · drop ${dr} · sfx ${sx}`;
}
// the counts walk every row of the source: once per frame, not once per painted row
let statsQ = 0;
const statsSoon = () => { if (!statsQ) statsQ = requestAnimationFrame(() => { statsQ = 0; stats(); }); };
function render() {
  const s = srcSel.value, r = rows(), pages = Math.max(1, Math.ceil(r.length / PER));
  page = Math.min(page, pages - 1);
  const list = $("list"); list.innerHTML = "";
  for (const x of r.slice(page * PER, (page + 1) * PER)) {
    const key = s + ":" + x.id, v = store[key];
    const d = document.createElement("div");
    if (x.ro) {  // reference row: crop, label, page link only
      d.className = "row";
      d.innerHTML = `<div class="crop"><img loading="lazy" src="${url(x.img)}"></div>
        <div><div class="meta">${esc(x.id)}${x.page ? ` · <a href="${url(x.page)}" target="_blank">page</a>` : ""}
          box ${x.box.join(",")}</div><div class="orig" style="font-size:17px">${esc(x.text)}</div></div>`;
      list.appendChild(d); continue;
    }
    d.className = "row" + (v ? " " + v.status : "");
    d.innerHTML = `<div class="crop"><img loading="lazy" src="${url(x.img)}"${x.zoom ? ' class="up"' : ""}></div>
      <div><div class="meta">${esc(x.id)}${x.status ? " · " + x.status : ""} · run ${x.run}
        · sus ${x.sus.toFixed(1)}${x.why ? " (" + esc(x.why) + ")" : ""}${x.lp != null ? " · logprob " + x.lp.toFixed(3) : ""}
        ${x.page ? `· <a href="${url(x.page)}" target="_blank">page</a>` : ""} box ${x.box.join(",")}</div>
      <div class="orig">${marked(x.text, +minIn.value || 0)}</div>
      ${x.note ? `<div class="meta">${esc(x.note)}</div>` : ""}
      ${x.rd != null ? `<div class="meta">__READER__${x.rsfx ? " SFX:" : ""} (${x.conf.toFixed(2)})${x.rwhy ? " [" + x.rwhy + "]" : ""}: ${esc(x.rd || "—").replace(/\n/g, " ⏎ ")}${x.edge ? " · edge " + esc(x.edge) : ""}</div>` : ""}
      ${x.hy != null ? `<div class="meta">hayai v2.7${x.hyd ? ` (${x.hyd.toFixed(2)}${x.hyx ? ", script slip" : ""})` : ""}: ${esc(x.hy || "—")}</div>` : ""}
      ${x.twin_img ? `<div class="twin"><img loading="lazy" src="${x.twin_img}"><div class="meta">twin (${esc(x.twin_lang)}) v4: ${esc(x.twin || "—").replace(/\n/g, " ⏎ ")}</div></div>` : ""}
      ${x.ms ? `<div class="meta">main (OCR:) · the largest text</div>` : ""}
      <textarea class="mainta" rows="${Math.max(2, x.text.split("\n").length + 1)}"></textarea>
      <div class="subbox" hidden><div class="meta">sub (OCR2:) · same crop · one item per line, reading order</div>
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
      drb.classList.toggle("on", !!w && w.status === "drop"); drb.classList.toggle("drop", true); statsSoon(); };
    ta.oninput = () => { store[key] = ta.value === x.text ? {status: "ok", text: null}
                                     : {status: "edit", text: ta.value}; save(); paint(); };
    // ok = this row is reviewed with what the box holds now (edited or not)
    okb.onclick = () => { store[key] = ta.value === x.text ? {status: "ok", text: null}
                                     : {status: "edit", text: ta.value}; save(); paint(); };
    drb.onclick = () => { store[key] = {status: "drop", text: null}; save(); paint(); };
    // Tab: ok this row with what the box holds, then the next row's main box (the next page's
    // first row after the last). A Hangul syllable still being composed is committed first.
    ta.onkeydown = e => {
      if (e.key !== "Tab" || e.shiftKey || e.ctrlKey || e.altKey || e.metaKey) return;
      e.preventDefault();
      const go = () => {
        okb.onclick();
        const all = [...$("list").querySelectorAll(".mainta")], nx = all[all.indexOf(ta) + 1];
        if (nx) { nx.focus(); nx.setSelectionRange(nx.value.length, nx.value.length);
          nx.scrollIntoView({block: "center"}); }
        else if (page < pages - 1) { page++; render(); scrollTo(0, 0);
          const f = $("list").querySelector(".mainta"); if (f) f.focus(); }
      };
      if (e.isComposing || e.keyCode === 229) {
        ta.addEventListener("compositionend", () => setTimeout(go, 0), {once: true}); ta.blur();
      } else go();
    };
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
  saveNow();
  const lines = ["source\tid\tstatus\torig\tnew\tsfx\tsub"];
  const byId = {};
  for (const [s, rows] of Object.entries(DATA)) { const m = byId[s] = new Map(); for (const y of rows) m.set(y.id, y); }
  let skipped = 0;
  for (const k of new Set([...Object.keys(store), ...Object.keys(marks), ...Object.keys(subs), ...AMARK])) {
    const v = store[k] || {status: "", text: null};
    const i = k.indexOf(":"), s = k.slice(0, i), id = k.slice(i + 1);
    const x = byId[s] && byId[s].get(id); if (!x || x.ro) continue;
    if (!changed(k, x)) { skipped += (!!store[k] || !!marks[k] || k in subs); continue; }
    // sub column: the row's sub when the reviewer touched it ("-" = removed); a reviewed
    // row whose sub box shows the proposal untouched exports the proposal; else empty
    const sb = k in subs ? (subs[k].trim() ? subs[k].replace(/\\n/g, "\n") : "-")
             : (store[k] && store[k].status !== "drop" ? subOf(k, x) : "");
    const q = t => (t == null ? "" : t).replace(/\\/g, "\\\\").replace(/\t/g, "\\t").replace(/\n/g, "\\n");
    // a typed "\\n" counts as a line break, same as Enter
    const nt = v.status === "edit" ? v.text.replace(/\\n/g, "\n") : "";
    // sfx: 1 = marked, 0 = an applied mark taken off, empty = no mark either side
    const sf = marks[k] ? "1" : AMARK.has(k) ? "0" : "";
    lines.push([s, id, v.status, q(x.text), q(nt), sf, q(sb)].join("\t"));
  }
  $("export").textContent = `Export TSV (last: ${lines.length - 1} changed, ${skipped} already applied)`;
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
    if lang == "zh":
        for pat, name in ((KANA, "kana"), (HANGUL, "hangul")):
            if pat.search(t):
                s += 3
                why.append(name)
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


KANA_HANJA = re.compile(r"[ぁ-ゖァ-ヺ一-鿿]")


def hayai_fields(hk: dict, src: str, rid: str, text: str) -> dict:
    """``hy`` / ``hyd`` for a kozh or hito row from ``--hayai_kozh``'s reads, or nothing."""
    read = hk.get((src, rid))
    if read is None:
        return {}
    alien = HANGUL if src == "hito_v4_ja" else KANA_HANJA
    hyx = int(bool(alien.search(read)) and not alien.search(text))
    hyd = hayai_distance(read, text, "")
    out = dict(hy=read, hyd=hyd, hyx=hyx)
    if hyd:
        out["hyk"] = hayai_kind(read, text, "")
    return out


def hito_rows(d: Path, flags: dict, hk: dict) -> dict[str, list[dict]]:
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
                **hayai_fields(hk, f"hito_v4_{side}", r.id, r.text),
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


def furigana_rows(
    d: Path, path: Path | None, fix: Path | None = None
) -> dict[str, list[dict]]:
    """``furigana_find.py read``'s crops with ruby groups: the label as the manifest holds
    it now, the sub box
    prefilled with every group's non-empty read in reading order: one line per main
    column (right to left), a column's groups top down, joined by a space. ``sus`` counts
    the groups that read as something else or sit over no kanji (those rows first)."""
    if path is None or not path.exists():
        return {}
    fixed = agent_subs(fix)
    # the label as the manifest holds it now (the read file keeps the text at find time)
    df = pd.read_parquet(d / "manifest_sfxp_speech.parquet")
    now = {
        f"{b}_{int(p):03d}_{i}": t
        for b, p, i, t, s in zip(df.book, df.page, df.id, df.text, df.source)
        if s in ("sfxp_speech", "m109_lines")
    }
    rows = []
    for line in open(path, encoding="utf-8"):
        x = json.loads(line)
        gs = sorted(x["groups"], key=lambda g: (g["col"], g["y0"]))
        if not gs:
            continue
        x["text"] = now.get(f"{x['book']}_{int(x['page']):03d}_{x['id']}", x["text"])
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


def linebreak_proposals(d: Path) -> dict[str, dict]:
    """``crop_dataset.break_proposals`` as the page shows them: the proposed breaks
    (``round_dots`` ∘ ``ellipsis_fold``), the label now, the read distance (+100 when a
    dot run stays odd) and a note with the column reads."""
    out = {}
    for k, x in break_proposals(d).items():
        text = round_dots(ellipsis_fold(x["out"]))
        odd = odd_dots(text)
        reads = " ⏎ ".join(x["lines"])
        out[k] = dict(
            text=text,
            cur=x["cur"],
            lbs=round(x["dist"] + (100.0 if odd else 0.0), 3),
            note=f"line breaks proposed ({x['n_segs']} segments"
            + (f", dots {odd}" if odd else "")
            + f", read distance {x['dist']:.2f}) · label: {x['cur']} · column reads: {reads}",
        )
    return out


LINE_DOT_RUN = re.compile(r"[.．・･…‥]+")


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
                gate="unchecked",  # the gate status a drop writes
                flat=1,  # the TSV keeps a line break as a space
                img=(DOUJIN_FROZEN / "crops" / f"{r.row}.png").as_uri(),
                page=page.as_uri() if page.exists() else "",
                box=[int(v) for v in box],
            )
        )
    print(f"[review] doujin_sfx: {len(rows)} rows ({len(reads)} reads)", flush=True)
    return {"doujin_sfx": rows}


def k3_rows(reads_path: Path | None) -> dict[str, list[dict]]:
    """The K3 gate's rows as ``k3_ko`` / ``k3_zh`` (id = the TSV's ``id``), with
    ``eval_k3.py``'s per-row read when ``reads_path`` is given."""
    df = pd.read_csv(K3_LABELS, sep="\t", dtype=str, keep_default_na=False)
    reads = {}
    if reads_path:
        for line in open(reads_path, encoding="utf-8"):
            x = json.loads(line)
            reads[str(x["id"])] = x
    boxes = {}
    for line in open(GELNOTE / "manifest.jsonl", encoding="utf-8"):
        x = json.loads(line)
        boxes[x["id"]] = x["box"]
    out = {"k3_ko": [], "k3_zh": []}
    for r in df.itertuples():
        sus, why = suspicion(r.text, int(r.w), int(r.h), r.lang)
        x = reads.get(r.id)
        page = next(iter(sorted((K3_FROZEN / "pages").glob(f"{r.image_id}.*"))), None)
        note = " · ".join(
            t
            for t in (
                f"script {r.zh_script}" if r.zh_script else "",
                r.label_note,
                f"translation: {r.note_body}" if r.note_body else "",
            )
            if t
        )
        out[f"k3_{r.lang}"].append(
            dict(
                id=r.id,
                text=r.text,
                run=longest_run(r.text),
                sus=sus,
                why=why,
                lf=int("\n" in r.text),
                len=len(r.text.replace(" ", "")),
                status=r.status,
                note=note,
                rd=x["pred"] if x else None,
                conf=round(float(x["sim"]), 3) if x else 0.0,
                edge="",
                rsfx=False,
                rwhy="runaway" if x and x.get("runaway") else "",
                agree=bool(x and x["exact"]),  # eval_k3's exact: off "dis"
                gate="skip",
                img=(K3_FROZEN / r.path).as_uri(),
                page=page.as_uri() if page else "",
                box=[int(v) for v in boxes.get(r.id, [])],
            )
        )
    print(
        f"[review] k3: {len(out['k3_ko'])} KO, {len(out['k3_zh'])} ZH ({len(reads)} reads)",
        flush=True,
    )
    return out


def hayai_reads(path: Path | None) -> dict[str, dict]:
    """``hayai_read.py``'s JSON lines by box id (``<book>_<page>_<n>``)."""
    if path is None or not path.exists():
        return {}
    return {x["id"]: x for x in map(json.loads, open(path, encoding="utf-8"))}


_SMALL_KANA = str.maketrans(
    "ぁぃぅぇぉゃゅょゎゕゖァィゥェォャュョヮヵヶ",
    "あいうえおやゆよわかけアイウエオヤユヨワカケ",
)
_KANA = re.compile(r"[ぁ-ゖァ-ヺ]")
_NOT_KANA_LINE = re.compile(r"[^\sぁ-ゖァ-ヺー〜~!?！？・…‥.っッ゛゜\u3099\u309a]")


def kana_fold(s: str) -> str:
    """Katakana → hiragana, small kana → large."""
    s = s.translate(_SMALL_KANA)
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)


def short_kana(main: str) -> bool:
    """A main of one kana, or the same kana two or three times (``へへへ``, ``あぁ…``):
    ``っ`` / ``ー`` / punctuation do not count, nothing but kana and those."""
    if _NOT_KANA_LINE.search(main):
        return False
    core = [c for c in exact_key(main) if _KANA.match(c) and c not in "っッ"]
    return 1 <= len(core) <= 3 and len({kana_fold(c) for c in core}) == 1


_LOOSE = [
    (re.compile(r"[ー―─—‐－~〜-]+"), "ー"),
    (re.compile(r"…+"), "…"),
    (re.compile(r"!+"), "!"),
    (re.compile(r"\?+"), "?"),
    (re.compile(r"。"), ""),
]


def loose_key(s: str) -> str:
    """``exact_key`` with small kana = large (``ォ`` / ``オ``; ``っ`` stays), a run of
    dashes and waves as one ``ー`` (a drawn long sound reads as either), a run of dots,
    ``!`` or ``?`` as one; ``。`` dropped."""
    s = exact_key(s).translate(_SMALL_KANA)
    for rx, to in _LOOSE:
        s = rx.sub(to, s)
    return s


def hayai_distance(read: str, main: str, sub: str) -> float:
    """0 where hayai's read is the main, or the main and the sub's items in either order
    (whitespace-blind, :func:`loose_key`'s folds; :func:`kana_fold` on both when the main
    is :func:`short_kana`: a lone へ / ヘ looks the same in the crop); else 1 − the
    similarity to the main."""
    h, m, s = loose_key(read), loose_key(main), loose_key(sub)
    if short_kana(main):
        h, m, s = kana_fold(h), kana_fold(m), kana_fold(s)
    if h in (m, m + s, s + m):
        return 0.0
    return round(1.0 - SequenceMatcher(None, h, m, autojunk=False).ratio(), 3) or 0.001


def _marks(s: str) -> tuple[str, int]:
    """``s`` with its dakuten / handakuten taken off, and how many letters held one as a
    composed glyph (``ゔ``, ``じ``, ``ぱ``). A mark NFC leaves standing (``あ゙``, a voiced
    vowel drawn for a scream) does not count: hayai adds those where the crop has none."""
    marks = "\u3099\u309a"
    bare = "".join(c for c in unicodedata.normalize("NFD", s) if c not in marks)
    n = sum(
        c not in marks and unicodedata.normalize("NFD", c)[-1] in marks
        for c in unicodedata.normalize("NFC", s)
    )
    return unicodedata.normalize("NFC", bare), n


def _subseq(a: str, b: str) -> bool:
    """``a`` is ``b`` with characters left out."""
    it = iter(b)
    return len(a) < len(b) and all(c in it for c in a)


def hayai_kind(read: str, main: str, sub: str) -> int:
    """Where a disagreeing hayai read points (the ``hayai: label errors first`` sort):
    2 = at the label (the read is the label plus a dakuten / handakuten, the label's
    characters reordered, or the label plus text the empty sub lacks), 0 = at hayai (the
    read is the label short of a mark or of characters, or is the sub alone: a misread or
    text the box cuts), 1 = neither."""
    h, m, s = loose_key(read), loose_key(main), loose_key(sub)
    if not h or not m:
        return 1
    (hb, hn), (mb, mn) = _marks(h), _marks(m)
    if hb == mb:
        return 2 if hn > mn else 0 if hn < mn else 1
    if len(m) > 1 and sorted(h) == sorted(m):
        return 2
    # text the label lacks: new characters (not hayai repeating the label), and not a
    # runaway read
    extra = h.replace(m, "", 1)
    if (
        not s
        and len(m) > 1
        and m in h
        and len(extra) >= 2
        and set(extra) - set(m)
        and len(h) <= 3 * len(m) + 10
    ):
        return 2
    if _subseq(h, m) or (s and (h == s or h in s)):
        return 0
    return 1


def agent_votes(d: Path | None) -> dict[str, list[dict]]:
    """Every ``out.tsv`` under ``d`` (``label_agent.md``'s output, one folder per agent
    and batch): ``{id: [{main, sub, conf, note}, …]}``, ``\\n`` unescaped."""
    out: dict[str, list[dict]] = {}
    if not d or not d.is_dir():
        return out
    for f in sorted(d.glob("*/out.tsv")):
        df = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False, quoting=3)
        for r in df.itertuples():
            out.setdefault(r.id, []).append(
                dict(
                    main=r.main.replace("\\n", "\n").strip(),
                    sub=r.sub.replace("\\n", "\n").strip(),
                    conf=r.conf.strip().lower(),
                    note=r.note,
                )
            )
    return out


def vote(votes: list[dict]) -> tuple[str | None, str, float]:
    """The agents' majority ``(main, sub, uncertainty)``: main ``None`` when no two agree
    (whitespace-blind). Uncertainty 0 = every agent wrote the same main at ``high``; each
    extra distinct main adds 1, each ``mid`` 0.25 and each ``low`` 0.5."""
    key = lambda t: re.sub(r"\s", "", t)  # noqa: E731
    groups: dict[str, list[dict]] = {}
    for v in votes:
        groups.setdefault(key(v["main"]), []).append(v)
    best = max(groups.values(), key=len)
    main = best[0]["main"] if len(best) >= 2 else None
    subs = [v["sub"] for v in best]
    sub = max(set(subs), key=subs.count) if main is not None else ""
    unc = (len(groups) - 1) + sum(
        {"high": 0.0, "mid": 0.25}.get(v["conf"], 0.5) for v in votes
    )
    return main, sub, unc


def anima_rows(
    path: Path | None, applied: dict, agents: Path | None = None
) -> dict[str, list[dict]]:
    """``anima_lowconf``: ``anima_lowconf.py``'s crops, the main box holding the agents'
    majority read when two or more agree (``label_agent.md``, ``--anima_agents``), else
    the v7 read; the applied main and sub of ``anima_mainsub.tsv`` over both."""
    if not path:
        return {}
    votes = agent_votes(agents)
    res = REPO / "post_image_dataset/resized"
    rows = []
    for x in map(json.loads, open(path, encoding="utf-8")):
        main, sub, extra = x["read"], "", {}
        note = f"v7 read: {x['read']!r} · score {x['score']:.3f} · det {x['det']:.2f}"
        if vs := votes.get(x["id"]):
            m, sb, unc = vote(vs)
            if m is not None:
                main, sub = m, sb
            extra["unc"] = round(unc, 2)
            extra["hg"] = int(
                bool(HANGUL.search(x["read"] + "".join(v["main"] for v in vs)))
            )
            note += " · agents: " + " / ".join(
                f"{v['main'] or '∅'!r} ({v['conf']})" for v in vs
            )
            if m is None:
                note += " · no majority: the box holds the v7 read"
        if ap := applied["ms"].get(f"anima_lowconf:{x['id']}"):
            extra["ap"] = ap[0]
            note = f"applied {ap[0]} · " + note
            if ap[0] != "drop":
                main, sub = ap[1], ap[2]
        rows.append(
            dict(
                id=x["id"],
                text=main,
                run=0,
                sus=0.0,
                why="",
                lf=int("\n" in main),
                len=len(main),
                status="",
                note=note,
                sub=sub,
                part=0,
                ms=1,
                sc=x["score"],
                img=(path.parent / "crops" / f"{x['id']}.png").as_uri(),
                page=(res / x["image"]).as_uri(),
                box=x["box"],
                **extra,
            )
        )
    print(f"[review] anima_lowconf: {len(rows)} rows", flush=True)
    return {"anima_lowconf": rows}


def reader_split_rows(
    path: Path | None, applied: dict, min_conf: float = 0.9
) -> dict[str, list[dict]]:
    """``reader_split``: cjk_rl's ``split_1126_0910`` crops (``reader_split.py``), kana renders
    one reader reads as the asked glyph and the other does not, kept where both readers'
    confidence is ``min_conf`` or over (the user, 11-26: both sure, one of them wrong). The main box holds the read
    that matched (hayai's on a ``hayai_only`` crop, the tuned reader's on a ``tuned_only``
    one); the meta line the asked glyph and both reads. ``key`` is what the caption asked
    for, not a checked label. The applied main and sub of ``reader_split_mainsub.tsv`` over
    both."""
    if not path:
        return {}
    rows = []
    for x in map(json.loads, open(path, encoding="utf-8")):
        rid = Path(x["file"]).stem
        confs = (float(x["hayai_conf"] or 0), float(x["tuned_conf"] or 0))
        if min(confs) < min_conf:
            continue
        top = max(confs)
        hy_won = x["case"] == "hayai_only"
        main, sub, extra = (x["hayai"] if hy_won else x["tuned"]) or "", "", {}
        note = (
            f"asked {x['key']} · {x['case']}{' · look-alike' if x['near'] else ''}"
            f" · tuned {x['tuned'] or '∅'!r} ({x['tuned_conf'] or 0:.2f})"
            f" · hayai {x['hayai'] or '∅'!r} ({x['hayai_conf'] or 0:.2f})"
            f" · {x['run']} / {x['arm']}"
        )
        if ap := applied["ms"].get(f"reader_split:{rid}"):
            extra["ap"] = ap[0]
            note = f"applied {ap[0]} · " + note
            if ap[0] != "drop":
                main, sub = ap[1], ap[2]
        rows.append(
            dict(
                id=rid,
                text=main,
                run=0,
                sus=0.0,
                why="",
                lf=int("\n" in main),
                len=len(main),
                status="",
                note=note,
                sub=sub,
                part=0,
                ms=1,
                zoom=1,
                # ks sort: hayai_only first, the asked glyph, the surer reader's confidence
                ks=[int(not hy_won), x["key"], -top],
                img=(path.parent / x["file"]).as_uri(),
                page=Path(x["page"]).as_uri(),
                box=x["box"],
                **extra,
            )
        )
    print(
        f"[review] reader_split: {len(rows)} rows (both conf >= {min_conf})",
        flush=True,
    )
    return {"reader_split": rows}


def mainsub_rows(
    d: Path,
    path: Path | None,
    ruby: list[dict] | None = None,
    hayai: dict[str, dict] | None = None,
    applied: dict | None = None,
) -> dict[str, list[dict]]:
    """``m109_mainsub.py``'s crops, one source each, every row. A row in
    ``m109_mainsub.tsv`` (``applied["ms"]``) shows the applied main and sub, not the
    proposal (``ap`` = its status; the page counts it reviewed), and hayai is measured
    against them; an applied drop has no hayai distance. A crop holding
    ``<text>`` boxes the furigana source holds (``sp``) takes their ruby reads (the
    agent's where it reviewed the row) as the head of its sub proposal; with one such box,
    ``fk`` points the page at that furigana row's sub box state.
    Image and page paths are relative to the Manga109 folder (``__M109__`` on the page).
    ``sus`` puts partly visible text, then a main that moved off the crop's label, first."""
    if path is None or not path.exists():
        return {}
    base = d.parent
    rel = lambda p: os.path.relpath(p, base)  # noqa: E731
    rb = {r["id"]: r for r in ruby or []}
    lb = linebreak_proposals(d)
    filled = filled_breaks()
    by: dict[str, list[dict]] = {}
    for line in open(path, encoding="utf-8"):
        x = json.loads(line)
        by.setdefault(x["source"], []).append(x)
    out = {}
    for src, xs in by.items():
        rows, n_rb, n_lb = [], 0, 0
        for x in xs:
            # a neg_speech box had no label to move off (it was an SFX: negative)
            moved = x["cat"] != "neg_speech" and x["main"] != x["label"]
            items = " | ".join(
                f"{i['k']} {i['t']} (g {i['g']:.0f})".replace("\n", " ⏎ ")
                for i in x["items"]
            )
            part = " | ".join(
                f"{p['k']} {p['t']} ({p['f']:.0%})".replace("\n", " ⏎ ")
                for p in x["part"]
            )
            note = f"in crop: {items or '—'}"
            if x.get("trim"):
                note = "main = a box only partly inside, cut it to what shows · " + note
            if part:
                note += f" · partly visible: {part}"
            if moved:
                lab = x["label"] or "(SFX: negative)"
                note += f" · label now: {lab}".replace("\n", " ⏎ ")
            sub, extra = x["sub"], {"rb": 0, "rsus": 0.0}
            keys = [f"{x['book']}_{int(x['page']):03d}_{t}" for t in x.get("sp", [])]
            main = with_breaks(x, filled)
            if main != x["main"]:
                note = "line breaks filled by an agent (trained as shown) · " + note
            # the main is one <text> box whose automatic breaks missed the gate: the
            # box holds the proposal (ok takes it), the meta line the label and reads
            # (the loader keeps these mains out of training until reviewed)
            if k := waiting_break(x, lb):
                lp = lb[k]
                extra["raw"] = main  # what trains until the row is reviewed
                main, extra["lbs"] = lp["text"], lp["lbs"]
                note = lp["note"] + " · " + note
                n_lb += 1
            hits = [rb[k] for k in keys if k in rb and rb[k]["sub"]]
            if "trim_ruby" in x:
                # a trim crop shows part of its box: the ruby inside the crop only
                hits = []
                if x["trim_ruby"]:
                    extra["rb"] = 1
                    sub = "\n".join([x["trim_ruby"]] + ([sub] if sub else []))
                    note = "ruby inside the crop · " + note
                    n_rb += 1
            if hits:
                extra["rb"] = 1
                extra["rsus"] = sum(r["sus"] for r in hits)
                sub = "\n".join([r["sub"] for r in hits] + ([sub] if sub else []))
                note = " · ".join(r["note"] for r in hits) + " · " + note
                if len(hits) == 1:
                    extra["fk"] = "furigana:" + hits[0]["id"]
                n_rb += 1
            if ap := ((applied or {}).get("ms") or {}).get(f"{src}:{x['id']}"):
                for f in ("raw", "lbs", "fk"):
                    extra.pop(f, None)
                note = f"applied {ap[0]} · " + note
                extra["ap"] = ap[0]
                if ap[0] != "drop":
                    main, sub = ap[1], ap[2]
            if (
                (hy := (hayai or {}).get(x["id"]))
                and hy["read"] is not None
                and extra.get("ap") != "drop"
            ):
                extra["hy"] = hy["read"]
                extra["hyd"] = hayai_distance(hy["read"], extra.get("raw", main), sub)
                if extra["hyd"]:
                    extra["hyk"] = hayai_kind(hy["read"], extra.get("raw", main), sub)
            rows.append(
                dict(
                    id=x["id"],
                    text=main,
                    run=0,
                    sus=2.0 * len(x["part"])
                    + 1.0 * moved
                    + 0.5 * bool(sub)
                    + 3.0 * x.get("trim", 0),
                    why="",
                    lf=int("\n" in main),
                    len=len(main),
                    status=x["cat"]
                    if x["split"] == "train"
                    else f"{x['split']} · {x['cat']}",
                    note=note,
                    sub=sub,
                    part=len(x["part"]),
                    ms=1,
                    img=rel(d / x["path"]),
                    page=rel(m109.page_path(x["book"], int(x["page"]))),
                    box=x["box"],
                    **extra,
                )
            )
        out[src] = rows
        print(
            f"[review] {src}: {len(rows)} rows "
            f"({sum(bool(r['sub']) for r in rows)} with a sub proposal, {n_rb} from ruby; "
            f"{n_lb} with proposed line breaks; "
            f"{sum(r['part'] > 0 for r in rows)} with partly visible text)",
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


def applied_state() -> dict:
    """What ``apply_review.py`` has written to the assets: SFX marks (``source:id``
    keys), subs (``sub_labels.tsv``) and Manga109 main / sub rows (``m109_mainsub.tsv``:
    [status, main, sub]). The page exports only the rows whose state differs from it."""
    A = ASSETS

    def rd(name: str) -> pd.DataFrame | None:
        p = A / name
        if not p.exists():
            return None
        return pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, quoting=3)

    un = lambda t: t.replace("\\n", "\n")  # noqa: E731
    out: dict = {"marks": [], "subs": {}, "ms": {}}
    if (m := rd("sfx_marks.tsv")) is not None:
        out["marks"] = sorted(set(m.source + ":" + m.id))
    if (s := rd("sub_labels.tsv")) is not None:
        out["subs"] = {f"{r.source}:{r.id}": un(r.sub) for r in s.itertuples()}
    for name in ("m109_mainsub.tsv", "anima_mainsub.tsv", "reader_split_mainsub.tsv"):
        if (ms := rd(name)) is not None:
            out["ms"].update(
                {
                    f"{r.source}:{r.id}": [r.status, un(r.main), un(r.sub)]
                    for r in ms.itertuples()
                }
            )
    print(
        f"[review] applied: {len(out['marks'])} marks, {len(out['subs'])} subs, "
        f"{len(out['ms'])} main / sub rows",
        flush=True,
    )
    return out


def mark_over_cap(data: dict[str, list[dict]], applied: dict) -> None:
    """``over`` on the rows whose main, as training takes it (the applied review, else
    the label before a line-break proposal), ``crop_dataset.cap_targets`` drops."""
    for src, rows in data.items():
        n = 0
        for r in rows:
            t = r.pop("raw", r["text"])
            if r.get("gate") or r.get("ro"):
                continue
            if v := applied["ms"].get(f"{src}:{r['id']}"):
                t = v[1]
            if len(normalize_target(t)) > MAX_TARGET_CHARS:
                r["over"] = 1
                n += 1
        if n:
            print(f"[review] {src}: {n} rows over the cap, not listed", flush=True)


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
        default=0,
        help="Manga109 reference rows per kind (m109_sfx / m109_speech); 0 = none (default)",
    )
    ap.add_argument(
        "--doujin_reads",
        type=Path,
        help="eval_sfx.py's output/ocr/eval/sfx_<name>.jsonl: a reader's read per "
        "doujin_sfx row",
    )
    ap.add_argument(
        "--k3_reads",
        type=Path,
        help="eval_k3.py's output/ocr/eval/k3_<name>.jsonl: a reader's read per "
        "k3_ko / k3_zh row",
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
        "--hayai",
        type=Path,
        help="hayai_read.py's output/ocr/hayai/<name>.jsonl: hayai's read per detector "
        "box (shown on the mainsub rows)",
    )
    ap.add_argument(
        "--hayai_kozh",
        type=Path,
        help="hayai_read.py --set kozh hito's output/ocr/hayai/<name>.jsonl: hayai's "
        "read per gelnote / swaprelax / hito crop (shown on those rows)",
    )
    ap.add_argument(
        "--mainsub",
        type=Path,
        help="m109_mainsub.py's output/ocr/mainsub/m109_mainsub.jsonl: the Manga109 "
        "main / sub sources (mainsub_det, mainsub_speech)",
    )
    ap.add_argument(
        "--anima_lowconf",
        type=Path,
        nargs="?",
        const=ANIMA_LOWCONF / "rows.jsonl",
        help="anima_lowconf.py's rows.jsonl (bare flag: the default root's): the Anima "
        "training set's low-confidence v7 reads",
    )
    ap.add_argument(
        "--anima_agents",
        type=Path,
        default=REPO / "output/ocr/anima_lowconf/agents",
        help="folder of label_agent.md outputs (*/out.tsv) for the anima_lowconf rows",
    )
    ap.add_argument(
        "--reader_split",
        type=Path,
        nargs="?",
        const=REPO / "output/cjk_rl/reader_split/split_1126_0910/manifest.jsonl",
        help="cjk_rl reader_split.py's manifest.jsonl (bare flag: split_1126_0910's): the "
        "kana crops the two readers split on",
    )
    ap.add_argument(
        "--reader_split_min_conf",
        type=float,
        default=0.9,
        help="list a reader_split crop only where both readers' confidence is this or over",
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
    hk = {}
    if a.hayai_kozh:
        for x in map(json.loads, open(a.hayai_kozh, encoding="utf-8")):
            if x["read"] is not None:
                hk[(x["source"], x["id"])] = x["read"]
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
                    **hayai_fields(hk, src, r.id, r.text),
                    img=(d / r.path).as_uri(),
                    page=Path(page).as_uri(),
                    box=[int(v) for v in box],
                )
            )
        data[src] = out
        print(f"[review] {src}: {len(out)} rows", flush=True)
    data.update(hito_rows(d, flags, hk))
    data.update(doujin_rows(a.doujin_reads))
    data.update(k3_rows(a.k3_reads))
    # the furigana rows are not a source of their own: they head the mainsub sub
    # proposals (and their earlier sub edits carry over through ``fk``)
    fur = furigana_rows(d, a.ruby, a.ruby_fix).get("furigana")
    hayai = hayai_reads(a.hayai)
    applied = applied_state()
    data.update(mainsub_rows(d, a.mainsub, fur, hayai, applied))
    data.update(anima_rows(a.anima_lowconf, applied, a.anima_agents))
    data.update(reader_split_rows(a.reader_split, applied, a.reader_split_min_conf))
    if a.ref_n:
        data.update(ref_rows(d, a.ref_n))
    mark_over_cap(data, applied)
    a.out.mkdir(parents=True, exist_ok=True)
    html = (
        PAGE.replace("__READER__", reader)
        .replace("__APPLIED__", json.dumps(applied, ensure_ascii=False))
        .replace("__M109__", json.dumps(d.parent.as_uri() + "/"))
        .replace("__DOUJIN_SORT__", "dis" if a.doujin_reads else "sus")
        .replace("__K3_SORT__", "dis" if a.k3_reads else "sus")
        .replace("__DATA__", json.dumps(data, ensure_ascii=False))
        .replace("__MINRUN__", json.dumps({s: a.min_run for s in data}))
    )
    (a.out / "index.html").write_text(html, encoding="utf-8")
    print(f"[review] → {a.out / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
