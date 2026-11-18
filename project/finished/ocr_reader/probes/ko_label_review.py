#!/usr/bin/env python3
"""A local review page for the Korean ``OCR:`` labels of the two-prompt reader
(``manifest_sfxp_kozh.parquet``: gelnote KO and swaprelax_v5) and the hito v4 self-labels
(``manifest_pseudo_hito_v4.parquet``, one source per side: ``hito_v4_ja`` / ``hito_v4_ko``):
the crop as trained, the label in an editable box, a status per row. Edits stay in the
browser (localStorage) and leave as a TSV through the page's Export button
(``apply_ko_review.py`` takes it).

Every editable row has an ``SFX`` checkbox (exported as the ``sfx`` column;
``apply_ko_review.py`` keeps the marks in ``assets/sfx_marks.tsv``, training does not read
them). ``m109_sfx`` / ``m109_speech`` are read-only references: a seeded sample of the
Manga109 COO onomatopoeia and the speech text (``manifest.parquet``), to compare against.

hito rows also show the twin crop (the other edition's lettering of the same region) with
its v4 read, and the read's mean token logprob (``prelabel_v4lp.jsonl``); they sort lowest
logprob first by default.

Every row is listed; the page filters on the longest spaceless Hangul run (the gelnote
teacher's dropped spacing passes the whitespace-blind agreement, ``pseudo_label.py``) and
sorts by it, by "no line break, longest first", or by ``suspicion`` (``suspicion``: the
label's line count against the crop's shape, kana / hanja, long spaceless runs — no reader
confidence is stored for either source).

    python project/finished/ocr_reader/probes/ko_label_review.py --out ~/ko_label_review
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ocr"))
import manga109 as m109  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
GELNOTE = Path("/media/sorryhyun/new/dataset/gelnote_crops")
SWAPRELAX = Path("/media/sorryhyun/new/dataset/hito_ocr/swap_relax_v5")
HITO = Path("/media/sorryhyun/new/dataset/hito_ocr/ko")
HANGUL = re.compile(r"[가-힣]")
RUN = re.compile(r"[가-힣]+")
KANA = re.compile(r"[\u3040-\u30ff]")
HANJA = re.compile(r"[\u4e00-\u9fff]")

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
    <option value="nolf">no line break, longest first</option>
    <option value="edge">edge text (label beyond v6's read)</option>
    <option value="dis">v6 disagrees, lowest confidence first</option>
    <option value="run">longest run first</option>
    <option value="id">id</option><option value="todo">unreviewed first</option></select></label>
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
const SORT_DEFAULT = {"gelnote_ko": "nolf", "swaprelax_v5": "sus", "hito_v4_ja": "lp", "hito_v4_ko": "lp"};
const PER = 50;
const KEY = "ko_label_review_v1";
let store = {};
try { store = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { store = {}; }
let marks = {};
try { marks = JSON.parse(localStorage.getItem(KEY + ":sfx") || "{}"); } catch (e) { marks = {}; }
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(store));
  localStorage.setItem(KEY + ":sfx", JSON.stringify(marks)); } catch (e) {} };
const $ = (id) => document.getElementById(id);
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
const srcSel = $("src"), minIn = $("minrun"), sortSel = $("sort");
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
minIn.onchange = sortSel.onchange = resort;
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
  else if (k === "lp") r = r.slice().sort((a, b) => (a.lp ?? 0) - (b.lp ?? 0) || a.id.localeCompare(b.id));
  else if (k === "sus") r = r.slice().sort((a, b) => b.sus - a.sus || a.id.localeCompare(b.id));
  else if (k === "edge") r = r.filter(x => x.edge).sort((a, b) => b.edge.length - a.edge.length);
  else if (k === "dis") r = r.filter(x => x.v6 != null && x.v6.replace(/\s/g, "") !== x.text.replace(/\s/g, ""))
                             .sort((a, b) => a.conf - b.conf);
  else if (k === "nolf") r = r.slice().sort((a, b) => (a.lf - b.lf) || (b.len - a.len));
  else if (k === "run") r = r.slice().sort((a, b) => b.run - a.run || a.id.localeCompare(b.id));
  else if (k === "id") r = r.slice().sort((a, b) => a.id.localeCompare(b.id));
  else r = r.slice().sort((a, b) => !!store[s + ":" + a.id] - !!store[s + ":" + b.id]);
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
      ${x.v6 != null ? `<div class="meta">v6 (${x.conf.toFixed(2)}): ${esc(x.v6).replace(/\n/g, " ⏎ ")}${x.edge ? " · edge " + esc(x.edge) : ""}</div>` : ""}
      ${x.twin_img ? `<div class="twin"><img loading="lazy" src="${x.twin_img}"><div class="meta">twin (${esc(x.twin_lang)}) v4: ${esc(x.twin || "—").replace(/\n/g, " ⏎ ")}</div></div>` : ""}
      <textarea rows="${Math.max(2, x.text.split("\n").length + 1)}"></textarea>
      <div class="btns"><button class="okb">✓ ok</button><button class="drb">✗ drop row</button>
        <button class="rsb">reset</button><label><input type="checkbox" class="sfxb"> SFX</label></div></div>`;
    const ta = d.querySelector("textarea");
    ta.value = v && v.text != null ? v.text : x.text;
    const okb = d.querySelector(".okb"), drb = d.querySelector(".drb");
    const sfxb = d.querySelector(".sfxb");
    sfxb.checked = !!marks[key];
    sfxb.onchange = () => { if (sfxb.checked) marks[key] = 1; else delete marks[key]; save(); paint(); };
    const paint = () => { const w = store[key];
      d.className = "row" + (w ? " " + w.status : "") + (marks[key] ? " sfx" : "");
      okb.classList.toggle("on", !!w && w.status === "ok"); okb.classList.toggle("ok", true);
      drb.classList.toggle("on", !!w && w.status === "drop"); drb.classList.toggle("drop", true); stats(); };
    ta.oninput = () => { store[key] = ta.value === x.text ? {status: "ok", text: null}
                                     : {status: "edit", text: ta.value}; save(); paint(); };
    // ok = this row is reviewed with what the box holds now (edited or not)
    okb.onclick = () => { store[key] = ta.value === x.text ? {status: "ok", text: null}
                                     : {status: "edit", text: ta.value}; save(); paint(); };
    drb.onclick = () => { store[key] = {status: "drop", text: null}; save(); paint(); };
    d.querySelector(".rsb").onclick = () => { delete store[key]; ta.value = x.text; save(); paint(); };
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
  const lines = ["source\tid\tstatus\torig\tnew\tsfx"];
  for (const k of new Set([...Object.keys(store), ...Object.keys(marks)])) {
    const v = store[k] || {status: "", text: null};
    const i = k.indexOf(":"), s = k.slice(0, i), id = k.slice(i + 1);
    const x = (DATA[s] || []).find(y => y.id === id); if (!x) continue;
    const q = t => (t == null ? "" : t).replace(/\\/g, "\\\\").replace(/\t/g, "\\t").replace(/\n/g, "\\n");
    // a typed "\\n" counts as a line break, same as Enter
    const nt = v.status === "edit" ? v.text.replace(/\\n/g, "\n") : "";
    lines.push([s, id, v.status, q(x.text), q(nt), marks[k] ? "1" : ""].join("\t"));
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


def hito_rows(d: Path) -> dict[str, list[dict]]:
    """``manifest_pseudo_hito_v4`` as two sources (``hito_v4_ja`` / ``hito_v4_ko``), each
    row with its twin crop, the twin's v4 read and the read's logprob."""
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
                v6=None,
                conf=0.0,
                edge="",
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
        help="ko_label_fix.py's <reads>_flags.parquet: v6 read, confidence, edge text per row",
    )
    ap.add_argument(
        "--ref_n",
        type=int,
        default=3000,
        help="Manga109 reference rows per kind (m109_sfx / m109_speech); 0 = none",
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
    flags = {}
    if a.flags:
        for f in pd.read_parquet(a.flags).itertuples():
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
                    v6=f.read if f is not None else None,
                    conf=round(float(f.conf), 3)
                    if f is not None and f.conf == f.conf
                    else 0.0,
                    edge=f.edge if f is not None and f.edge else "",
                    img=(d / r.path).as_uri(),
                    page=Path(page).as_uri(),
                    box=[int(v) for v in box],
                )
            )
        data[src] = out
        print(f"[review] {src}: {len(out)} rows", flush=True)
    data.update(hito_rows(d))
    if a.ref_n:
        data.update(ref_rows(d, a.ref_n))
    a.out.mkdir(parents=True, exist_ok=True)
    html = PAGE.replace("__DATA__", json.dumps(data, ensure_ascii=False)).replace(
        "__MINRUN__", json.dumps({s: a.min_run for s in data})
    )
    (a.out / "index.html").write_text(html, encoding="utf-8")
    print(f"[review] → {a.out / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
