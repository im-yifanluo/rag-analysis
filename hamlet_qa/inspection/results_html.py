"""Render Hamlet QA result JSON/JSONL as an interactive static HTML viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hamlet_qa.core.io import load_jsonl
from hamlet_qa.inspection.read_results import load_result_rows


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Hamlet QA Results</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f8;
      --panel: #ffffff;
      --panel-soft: #f0f3f5;
      --text: #151a1f;
      --muted: #65717d;
      --line: #d8dee4;
      --accent: #0f766e;
      --accent-strong: #115e59;
      --good: #1f7a3f;
      --bad: #b42318;
      --warn: #9a5b00;
      --code: #f8fafb;
      --shadow: 0 1px 2px rgba(18, 27, 35, 0.08);
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 15px;
      line-height: 1.45;
    }

    button,
    input,
    select {
      font: inherit;
    }

    .app {
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
    }

    header {
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      padding: 18px 24px 16px;
      position: sticky;
      top: 0;
      z-index: 10;
      box-shadow: var(--shadow);
    }

    .topline {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 14px;
    }

    h1 {
      margin: 0;
      font-size: 22px;
      font-weight: 720;
      letter-spacing: 0;
    }

    .source {
      color: var(--muted);
      font-size: 13px;
      word-break: break-word;
    }

    .controls {
      display: grid;
      grid-template-columns: minmax(220px, 1fr) minmax(150px, 210px) minmax(150px, 210px) minmax(150px, 210px) minmax(150px, 210px) auto;
      gap: 10px;
      align-items: end;
    }

    label {
      display: grid;
      gap: 4px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }

    input[type="search"],
    input[type="file"],
    select {
      min-height: 38px;
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      padding: 8px 10px;
    }

    .toggle {
      min-height: 38px;
      display: flex;
      align-items: center;
      gap: 7px;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      font-size: 13px;
      font-weight: 650;
      white-space: nowrap;
    }

    main {
      width: 100%;
      max-width: 1680px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: clamp(170px, 15vw, 220px) minmax(0, 1fr);
      gap: 14px;
      padding: 18px 24px 40px;
      align-items: start;
    }

    aside {
      min-width: 0;
      align-self: start;
      position: sticky;
      top: 132px;
      display: grid;
      gap: 10px;
    }

    .panel {
      min-width: 0;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }

    .panel-pad {
      padding: 10px;
    }

    .stats {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 8px;
    }

    .stat {
      background: var(--panel-soft);
      border-radius: 6px;
      padding: 8px;
      min-width: 0;
    }

    .stat strong {
      display: block;
      font-size: 18px;
      line-height: 1.1;
    }

    .stat span {
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }

    .question-nav {
      max-height: calc(100vh - 330px);
      overflow: auto;
    }

    .nav-item {
      width: 100%;
      border: 0;
      border-bottom: 1px solid var(--line);
      background: transparent;
      padding: 9px 10px;
      text-align: left;
      cursor: pointer;
      min-width: 0;
    }

    .nav-item:hover,
    .nav-item.active {
      background: #e7f2f0;
    }

    .nav-item strong {
      display: block;
      font-size: 12px;
      margin-bottom: 3px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .nav-item span {
      display: block;
      color: var(--muted);
      font-size: 11px;
      overflow: hidden;
      text-overflow: ellipsis;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
    }

    .content {
      min-width: 0;
      display: grid;
      gap: 16px;
    }

    .summary-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }

    #summaryPanel {
      overflow-x: auto;
    }

    .summary-table th,
    .summary-table td {
      border-bottom: 1px solid var(--line);
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
    }

    .summary-table th {
      color: var(--muted);
      font-size: 12px;
      font-weight: 720;
      background: var(--panel-soft);
    }

    .summary-table td.num,
    .summary-table th.num {
      text-align: right;
      font-variant-numeric: tabular-nums;
    }

    .question-card {
      min-width: 0;
      overflow: hidden;
    }

    .question-head {
      padding: 16px;
      border-bottom: 1px solid var(--line);
      display: grid;
      gap: 10px;
    }

    .qid {
      color: var(--accent-strong);
      font-size: 13px;
      font-weight: 760;
      word-break: break-word;
    }

    h2 {
      margin: 0;
      font-size: 18px;
      line-height: 1.3;
      letter-spacing: 0;
    }

    .expected {
      margin: 0;
      color: var(--muted);
    }

    .treatments {
      display: grid;
      gap: 12px;
      padding: 14px;
    }

    .result-card {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      overflow: hidden;
    }

    .result-head {
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfd;
      display: grid;
      grid-template-columns: minmax(180px, 1fr) auto;
      gap: 10px;
      align-items: center;
    }

    .treatment-name {
      font-weight: 760;
      word-break: break-word;
    }

    .metrics {
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 6px;
    }

    .badge {
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      border-radius: 999px;
      padding: 3px 8px;
      background: var(--panel-soft);
      color: var(--muted);
      font-size: 12px;
      font-weight: 680;
      white-space: nowrap;
    }

    .badge.good {
      color: var(--good);
      background: #eaf7ef;
    }

    .badge.bad {
      color: var(--bad);
      background: #fff0ee;
    }

    .badge.warn {
      color: var(--warn);
      background: #fff6df;
    }

    .badge.judge {
      font-size: 13px;
      font-weight: 760;
      border: 1px solid currentColor;
    }

    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }

    .chip {
      display: inline-flex;
      align-items: baseline;
      gap: 5px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel-soft);
      padding: 3px 8px;
      font-size: 12px;
      white-space: nowrap;
    }

    .chip b {
      color: var(--muted);
      font-weight: 680;
    }

    .mono {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      font-size: 12.5px;
      overflow-wrap: anywhere;
    }

    .judge-rationale {
      border-left: 3px solid var(--accent);
      padding: 8px 10px;
      background: var(--panel-soft);
      border-radius: 0 6px 6px 0;
    }

    .slot-ok {
      color: var(--good);
      font-weight: 720;
    }

    .slot-miss {
      color: var(--bad);
      font-weight: 720;
    }

    .metric-annotations li.ci-positive {
      color: var(--good);
    }

    .metric-annotations li.ci-negative {
      color: var(--bad);
    }

    .result-body {
      display: grid;
      gap: 14px;
      padding: 14px;
    }

    h3 {
      margin: 0 0 8px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 760;
      text-transform: uppercase;
      letter-spacing: 0;
    }

    pre,
    .text-block {
      margin: 0;
      padding: 12px;
      background: var(--code);
      border: 1px solid var(--line);
      border-radius: 6px;
      color: var(--text);
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      font-size: 13px;
      line-height: 1.5;
    }

    .text-block {
      font-family: inherit;
    }

    .two-col {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 12px;
    }

    .list {
      display: grid;
      gap: 8px;
    }

    .list-row {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      background: #fff;
    }

    .list-row .meta {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 5px;
      word-break: break-word;
    }

    details {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      overflow: hidden;
    }

    summary {
      cursor: pointer;
      padding: 10px 12px;
      font-weight: 720;
      background: #fbfcfd;
      border-bottom: 1px solid transparent;
      word-break: break-word;
    }

    details[open] summary {
      border-bottom-color: var(--line);
    }

    .details-body {
      padding: 12px;
      display: grid;
      gap: 10px;
    }

    .chunk-meta {
      color: var(--muted);
      font-size: 12px;
    }

    .empty {
      padding: 18px;
      color: var(--muted);
      text-align: center;
    }

    .hidden {
      display: none;
    }

    @media (max-width: 1180px) {
      header {
        position: static;
      }

      .controls {
        grid-template-columns: 1fr 1fr;
      }

      main {
        grid-template-columns: 1fr;
      }

      aside {
        position: static;
      }

      .question-nav {
        max-height: 220px;
      }

      .stats {
        grid-template-columns: repeat(4, minmax(0, 1fr));
      }
    }

    @media (max-width: 680px) {
      header,
      main {
        padding-left: 12px;
        padding-right: 12px;
      }

      .topline,
      .result-head {
        grid-template-columns: 1fr;
        display: grid;
      }

      .controls,
      .two-col {
        grid-template-columns: 1fr;
      }

      .metrics {
        justify-content: flex-start;
      }
    }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <div class="topline">
        <div>
          <h1>Hamlet QA Results</h1>
          <div class="source" id="sourceLabel"></div>
        </div>
        <div class="source" id="rowLabel"></div>
      </div>
      <div class="controls">
        <label>
          Search
          <input id="searchInput" type="search" placeholder="Question, output, chunk id">
        </label>
        <label>
          Treatment
          <select id="treatmentSelect"></select>
        </label>
        <label>
          Question
          <select id="questionSelect"></select>
        </label>
        <label>
          Results File
          <input id="resultsFileInput" type="file" accept=".json,.jsonl,application/json">
        </label>
        <label>
          Chunks File
          <input id="chunksFileInput" type="file" accept=".json,.jsonl,application/json">
        </label>
        <label class="toggle">
          <input id="promptToggle" type="checkbox">
          Open prompts
        </label>
      </div>
    </header>
    <main>
      <aside>
        <section class="panel panel-pad">
          <div class="stats" id="stats"></div>
        </section>
        <section class="panel">
          <div class="question-nav" id="questionNav"></div>
        </section>
      </aside>
      <section class="content">
        <section class="panel" id="summaryPanel"></section>
        <section id="results"></section>
      </section>
    </main>
  </div>
  <script type="application/json" id="embedded-results">__RESULT_DATA__</script>
  <script type="application/json" id="embedded-chunks">__CHUNK_DATA__</script>
  <script>
    const embeddedSource = __SOURCE_LABEL__;
    const embeddedChunkSource = __CHUNK_SOURCE_LABEL__;
    let rows = [];
    let chunksById = {};
    let currentResultSource = embeddedSource;
    let currentChunkSource = embeddedChunkSource;
    let selectedQuestion = "all";

    const els = {
      sourceLabel: document.getElementById("sourceLabel"),
      rowLabel: document.getElementById("rowLabel"),
      searchInput: document.getElementById("searchInput"),
      treatmentSelect: document.getElementById("treatmentSelect"),
      questionSelect: document.getElementById("questionSelect"),
      resultsFileInput: document.getElementById("resultsFileInput"),
      chunksFileInput: document.getElementById("chunksFileInput"),
      promptToggle: document.getElementById("promptToggle"),
      stats: document.getElementById("stats"),
      questionNav: document.getElementById("questionNav"),
      summaryPanel: document.getElementById("summaryPanel"),
      results: document.getElementById("results"),
    };

    function parseRows(text) {
      const trimmed = text.trim();
      if (!trimmed) return [];
      try {
        const parsed = JSON.parse(trimmed);
        if (Array.isArray(parsed)) return parsed;
        if (parsed && typeof parsed === "object") return [parsed];
        throw new Error("Expected a JSON object or array.");
      } catch (jsonError) {
        const parsedLines = [];
        const lines = trimmed.split(/\\r?\\n/);
        for (let index = 0; index < lines.length; index += 1) {
          const line = lines[index].trim();
          if (!line) continue;
          try {
            const row = JSON.parse(line);
            if (!row || typeof row !== "object" || Array.isArray(row)) {
              throw new Error("line is not an object");
            }
            parsedLines.push(row);
          } catch (lineError) {
            throw new Error(`Could not parse JSON/JSONL at line ${index + 1}: ${lineError.message}`);
          }
        }
        return parsedLines;
      }
    }

    function normalizeChunks(data) {
      const map = {};
      const values = Array.isArray(data) ? data : Object.values(data || {});
      for (const chunk of values) {
        if (chunk && typeof chunk === "object" && chunk.chunk_id) {
          map[chunk.chunk_id] = chunk;
        }
      }
      return map;
    }

    function parseChunks(text) {
      const trimmed = text.trim();
      if (!trimmed) return {};
      try {
        return normalizeChunks(JSON.parse(trimmed));
      } catch (jsonError) {
        return normalizeChunks(parseRows(trimmed));
      }
    }

    function updateSourceLabel() {
      const chunkText = currentChunkSource ? currentChunkSource : "chunks not loaded";
      els.sourceLabel.textContent = `${currentResultSource} | chunks: ${chunkText}`;
    }

    function loadEmbeddedData() {
      const rowsNode = document.getElementById("embedded-results");
      const chunksNode = document.getElementById("embedded-chunks");
      rows = parseRows(rowsNode.textContent || "[]");
      chunksById = normalizeChunks(JSON.parse(chunksNode.textContent || "{}"));
      updateSourceLabel();
    }

    function unique(values) {
      return Array.from(new Set(values)).filter(Boolean);
    }

    function numberValue(value) {
      if (value === null || value === undefined || value === "n/a") return null;
      const parsed = Number(value);
      return Number.isFinite(parsed) ? parsed : null;
    }

    function fmt(value) {
      if (value === null || value === undefined) return "n/a";
      if (typeof value === "number") {
        return Number.isInteger(value) ? String(value) : value.toFixed(3);
      }
      return String(value);
    }

    function score(value) {
      const parsed = numberValue(value);
      return parsed === null ? "n/a" : parsed.toFixed(6);
    }

    function scoreDetails(hit) {
      const parts = [`score ${score(hit.score)}`];
      [
        ["dense_rank", "dense_rank"],
        ["dense_score", "dense_score"],
        ["rerank_score", "rerank_score"],
        ["sparse_rank", "sparse_rank"],
        ["sparse_score", "sparse_score"],
        ["method", "retrieval_method"],
      ].forEach(([label, key]) => {
        if (hit[key] !== undefined) parts.push(`${label} ${fmt(hit[key])}`);
      });
      return parts.join("; ");
    }

    function mean(values) {
      const numeric = values.map(numberValue).filter((value) => value !== null);
      if (!numeric.length) return null;
      return numeric.reduce((total, value) => total + value, 0) / numeric.length;
    }

    function text(value) {
      if (value === null || value === undefined) return "";
      return String(value);
    }

    function el(tag, className, content) {
      const node = document.createElement(tag);
      if (className) node.className = className;
      if (content !== undefined) node.textContent = content;
      return node;
    }

    function badge(label, value, kind) {
      const node = el("span", `badge ${kind || ""}`.trim(), `${label}: ${fmt(value)}`);
      return node;
    }

    function recallKind(value) {
      const numeric = numberValue(value);
      if (numeric === null) return "";
      if (numeric >= 1) return "good";
      if (numeric <= 0) return "bad";
      return "warn";
    }

    function rowSearchText(row) {
      const chunkIds = (row.selected_chunk_ids || []).join(" ");
      const evidenceIds = evidenceChunkIds(row).join(" ");
      const quotes = (row.required_quotes_present_in_context || row.required_evidence_quotes || [])
        .map((quote) => quote.quote || "")
        .join(" ");
      return [
        row.question_id,
        row.treatment,
        row.question,
        row.expected_answer,
        row.model_output,
        row.judge_rationale,
        chunkIds,
        evidenceIds,
        quotes,
      ].map(text).join(" ").toLowerCase();
    }

    function filteredRows() {
      const query = els.searchInput.value.trim().toLowerCase();
      const treatment = els.treatmentSelect.value;
      const question = els.questionSelect.value;
      return rows.filter((row) => {
        if (treatment !== "all" && row.treatment !== treatment) return false;
        if (question !== "all" && row.question_id !== question) return false;
        if (query && !rowSearchText(row).includes(query)) return false;
        return true;
      });
    }

    function groupByQuestion(activeRows) {
      const groups = new Map();
      for (const row of activeRows) {
        const key = row.question_id || "unknown_question";
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key).push(row);
      }
      return groups;
    }

    function renderStats(activeRows) {
      const questions = unique(activeRows.map((row) => row.question_id)).length;
      const treatments = unique(activeRows.map((row) => row.treatment)).length;
      const chunks = Object.keys(chunksById).length;
      const stats = [
        ["Rows", activeRows.length],
        ["Questions", questions],
        ["Treatments", treatments],
        ["Corpus Chunks", chunks],
      ];
      els.stats.replaceChildren(...stats.map(([label, value]) => {
        const box = el("div", "stat");
        box.append(el("strong", "", value));
        box.append(el("span", "", label));
        return box;
      }));
      els.rowLabel.textContent = `${activeRows.length} of ${rows.length} rows`;
    }

    function renderSummary(activeRows) {
      const byTreatment = new Map();
      for (const row of activeRows) {
        const key = row.treatment || "unknown";
        if (!byTreatment.has(key)) byTreatment.set(key, []);
        byTreatment.get(key).push(row);
      }

      const table = el("table", "summary-table");
      const thead = document.createElement("thead");
      const headRow = document.createElement("tr");
      ["Treatment", "Rows", "Mean judge rating", "Mean quote recall", "Mean chunk recall", "Mean context tokens", "Suff. context rate", "Mean CI+ fraction"].forEach((label, index) => {
        headRow.append(el("th", index === 0 ? "" : "num", label));
      });
      thead.append(headRow);
      table.append(thead);

      const tbody = document.createElement("tbody");
      Array.from(byTreatment.entries()).sort(([a], [b]) => a.localeCompare(b)).forEach(([treatment, treatmentRows]) => {
        const row = document.createElement("tr");
        const cells = [
          treatment,
          treatmentRows.length,
          mean(treatmentRows.map((item) => item.judge_rating)),
          mean(treatmentRows.map((item) => item.evidence_quote_recall)),
          mean(treatmentRows.map((item) => item.evidence_chunk_recall)),
          mean(treatmentRows.map((item) => item.context_tokens)),
          mean(treatmentRows.map((item) => item.sufficient_context)),
          mean(treatmentRows.map((item) => item.ci_positive_fraction)),
        ];
        cells.forEach((value, index) => row.append(el("td", index === 0 ? "" : "num", fmt(value))));
        tbody.append(row);
      });
      table.append(tbody);
      els.summaryPanel.replaceChildren(table);
    }

    function renderNav(activeRows) {
      const groups = groupByQuestion(activeRows);
      const buttons = [];
      for (const [questionId, groupRows] of groups.entries()) {
        const first = groupRows[0] || {};
        const button = el("button", `nav-item ${selectedQuestion === questionId ? "active" : ""}`.trim());
        button.type = "button";
        button.append(el("strong", "", questionId));
        button.append(el("span", "", first.question || ""));
        button.addEventListener("click", () => {
          selectedQuestion = questionId;
          document.getElementById(`q-${cssSafe(questionId)}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
          render();
        });
        buttons.push(button);
      }
      if (!buttons.length) {
        els.questionNav.replaceChildren(el("div", "empty", "No matching questions."));
      } else {
        els.questionNav.replaceChildren(...buttons);
      }
    }

    function cssSafe(value) {
      return String(value || "unknown").replace(/[^a-zA-Z0-9_-]/g, "_");
    }

    function renderPre(label, value) {
      const section = el("section", "");
      section.append(el("h3", "", label));
      section.append(el("pre", "", text(value) || "n/a"));
      return section;
    }

    function renderTextBlock(label, value) {
      const section = el("section", "");
      section.append(el("h3", "", label));
      section.append(el("div", "text-block", text(value) || "n/a"));
      return section;
    }

    function chunkForId(row, chunkId) {
      if (chunksById[chunkId]) return chunksById[chunkId];
      return (row.raw_chunks || []).find((chunk) => chunk.chunk_id === chunkId) || null;
    }

    function evidenceChunkIds(row) {
      const ids = [];
      const seen = new Set();
      const add = (chunkId) => {
        if (!chunkId || seen.has(chunkId)) return;
        seen.add(chunkId);
        ids.push(chunkId);
      };
      const quotes = row.required_quotes_present_in_context || row.required_evidence_quotes || [];
      for (const quote of quotes) {
        for (const chunkId of quote.matched_chunk_ids || []) {
          add(chunkId);
        }
      }
      for (const chunkId of row.derived_gold_chunk_ids || []) {
        add(chunkId);
      }
      return ids;
    }

    function renderChunkDetails(row, chunkId, missingLabel) {
      const chunk = chunkForId(row, chunkId);
      const details = el("details", "");
      if (!chunk) {
        details.append(el("summary", "", chunkId));
        const body = el("div", "details-body");
        body.append(el("div", "empty", missingLabel || "Chunk text was not found in the loaded chunk data."));
        details.append(body);
        return details;
      }
      const title = `${chunk.chunk_id} | Act ${chunk.act} Scene ${chunk.scene} | ${chunk.token_count} tokens`;
      details.append(el("summary", "", title));
      const body = el("div", "details-body");
      body.append(el("div", "chunk-meta", `${chunk.scene_title || ""} | global_index ${chunk.global_index} | tokens ${chunk.start_token}-${chunk.end_token}`));
      body.append(el("pre", "", chunk.text || ""));
      details.append(body);
      return details;
    }

    function renderQuotes(row) {
      const section = el("section", "");
      section.append(el("h3", "", "Evidence Quotes"));
      const list = el("div", "list");
      const quotes = row.required_quotes_present_in_context || row.required_evidence_quotes || [];
      if (!quotes.length) {
        list.append(el("div", "empty", "No required evidence quotes."));
      } else {
        for (const quote of quotes) {
          const item = el("div", "list-row");
          const present = quote.present === undefined ? "not recorded" : quote.present ? "present" : "missing";
          const matched = (quote.matched_chunk_ids || []).join(", ") || "none";
          item.append(el("div", "meta", `${present}; role: ${quote.role || "unknown"}; matched chunks: ${matched}`));
          item.append(el("div", "", quote.quote || ""));
          list.append(item);
        }
      }
      section.append(list);
      return section;
    }

    function renderEvidenceChunks(row) {
      const section = el("section", "");
      section.append(el("h3", "", "Evidence Chunks"));
      const ids = evidenceChunkIds(row);
      if (!ids.length) {
        section.append(el("div", "empty", "No evidence chunk IDs recorded."));
        return section;
      }
      const list = el("div", "list");
      for (const chunkId of ids) {
        list.append(renderChunkDetails(row, chunkId));
      }
      section.append(list);
      return section;
    }

    function renderIdLists(row) {
      const wrap = el("section", "two-col");
      const selected = renderSimpleList("Selected Chunk IDs", row.selected_chunk_ids || []);
      const gold = renderSimpleList("Derived Gold Chunk IDs", row.derived_gold_chunk_ids || []);
      wrap.append(selected, gold);
      return wrap;
    }

    function renderSimpleList(label, values) {
      const section = el("section", "");
      section.append(el("h3", "", label));
      const list = el("div", "list");
      if (!values.length) {
        list.append(el("div", "empty", "none"));
      } else {
        values.forEach((value) => list.append(el("div", "list-row", value)));
      }
      section.append(list);
      return section;
    }

    function renderRetrieval(row) {
      const section = el("section", "");
      section.append(el("h3", "", "Retrieval"));
      const details = el("details", "");
      const selectedScores = row.retrieval_scores || [];
      const trace = row.retrieval_trace || [];
      details.append(el("summary", "", `${selectedScores.length} selected scores, ${trace.length} trace hits`));
      const body = el("div", "details-body");
      body.append(renderSimpleList(
        "Scores For Selected Chunks",
        selectedScores.map((hit) => `rank ${hit.rank}: ${hit.chunk_id} (${scoreDetails(hit)})`)
      ));
      const traceRows = trace.map((hit) => {
        const location = `Act ${hit.act} Scene ${hit.scene}`;
        return `rank ${hit.rank}: ${hit.chunk_id} (${scoreDetails(hit)}; ${location}; global_index ${hit.global_index})`;
      });
      body.append(renderSimpleList("Full Retrieval Trace", traceRows));
      details.append(body);
      section.append(details);
      return section;
    }

    function renderChunks(row) {
      const section = el("section", "");
      section.append(el("h3", "", "Selected Context Chunks"));
      const chunks = row.raw_chunks || [];
      if (!chunks.length) {
        section.append(el("div", "empty", "No selected context chunks."));
        return section;
      }
      const list = el("div", "list");
      for (const chunk of chunks) {
        list.append(renderChunkDetails(row, chunk.chunk_id));
      }
      section.append(list);
      return section;
    }

    function renderPrompts(row) {
      const section = el("section", "");
      section.append(el("h3", "", "Raw Prompts"));
      const outer = el("details", "");
      outer.open = els.promptToggle.checked;
      outer.append(el("summary", "", "System, user, and full prompt"));
      const body = el("div", "details-body");
      for (const [label, value] of [
        ["System Prompt", row.system_prompt],
        ["User Prompt", row.user_prompt],
        ["Full Prompt", row.full_prompt],
      ]) {
        const details = el("details", "");
        details.open = els.promptToggle.checked;
        details.append(el("summary", "", label));
        const detailsBody = el("div", "details-body");
        detailsBody.append(el("pre", "", text(value) || "n/a"));
        details.append(detailsBody);
        body.append(details);
      }
      outer.append(body);
      section.append(outer);
      return section;
    }

    // Decomposition prompts were renamed to describe the cognitive instruction;
    // alias older recorded variant names so historical runs display the new ones.
    const PROMPT_ALIASES = {
      subquestions: "split_questions",
      info_requirements: "list_requirements",
      strategy: "reason_then_plan",
    };

    function friendlyPrompt(value) {
      if (value === null || value === undefined) return value;
      return PROMPT_ALIASES[value] || value;
    }

    function judgeKind(value) {
      const numeric = numberValue(value);
      if (numeric === null) return "";
      if (numeric >= 4) return "good";
      if (numeric >= 2.5) return "warn";
      return "bad";
    }

    function renderJudge(row) {
      const section = el("section", "");
      if (row.judge_rating === undefined || row.judge_rating === null) return section;
      section.append(el("h3", "", `LLM-Judge Rating: ${fmt(row.judge_rating)} / 5`));
      if (row.judge_rationale) {
        section.append(el("div", "judge-rationale", row.judge_rationale));
      }
      if (row.judge_rubric || row.judge_model) {
        const details = el("details", "");
        details.append(el("summary", "", `Judge: ${row.judge_model || "unknown"} (rubric)`));
        const body = el("div", "details-body");
        body.append(el("div", "chunk-meta", row.judge_rubric || "no rubric recorded"));
        details.append(body);
        section.append(details);
      }
      return section;
    }

    function chip(label, value) {
      const node = el("span", "chip");
      node.append(el("b", "", label));
      node.append(el("span", "", fmt(value)));
      return node;
    }

    function chipRow(pairs) {
      const wrap = el("div", "chips");
      pairs.forEach(([label, value]) => {
        if (value !== undefined && value !== null && value !== "") wrap.append(chip(label, value));
      });
      return wrap;
    }

    function renderPlanNodes(nodes) {
      const list = el("div", "list");
      if (!nodes || !nodes.length) {
        list.append(el("div", "empty", "No evidence nodes."));
        return list;
      }
      for (const node of nodes) {
        const item = el("div", "list-row");
        const deps = (node.depends_on || []).join(", ");
        item.append(el("div", "meta",
          `${node.node_id || "?"} | order ${fmt(node.order_index)}` + (deps ? ` | depends_on: ${deps}` : "")));
        item.append(el("div", "", node.need || ""));
        item.append(el("div", "mono chunk-meta", `query: ${node.node_query || ""}`));
        list.append(item);
      }
      return list;
    }

    function renderCollapsedPre(label, value, mono) {
      const details = el("details", "");
      details.append(el("summary", "", label));
      const body = el("div", "details-body");
      body.append(el("pre", "", text(value) || "n/a"));
      details.append(body);
      return details;
    }

    function renderPerNodeRetrieval(row, execution, nodes) {
      const wrap = el("div", "list");
      const perNode = execution.per_node_retrieval || [];
      const supportPerNode = ((execution.support_scoring || {}).per_node) || {};
      const needByNode = {};
      (nodes || []).forEach((node) => { needByNode[node.node_id] = node.need || ""; });
      if (!perNode.length) {
        wrap.append(el("div", "empty", "No per-node retrieval recorded."));
        return wrap;
      }
      for (const entry of perNode) {
        const nodeId = entry.node_id || "?";
        const hits = entry.retrieved || [];
        const supports = {};
        (supportPerNode[nodeId] || []).forEach((item) => { supports[item.chunk_id] = item.support; });
        const details = el("details", "");
        const reformTag = entry.reformulated ? " | query reformulated" : "";
        details.append(el("summary", "", `${nodeId} | ${hits.length} candidates${reformTag}`));
        const body = el("div", "details-body");
        if (needByNode[nodeId]) body.append(el("div", "chunk-meta", `need: ${needByNode[nodeId]}`));
        body.append(el("div", "mono chunk-meta", `query used: ${entry.query_used || ""}`));
        if (entry.reformulated && entry.reformulated.query) {
          body.append(el("div", "mono chunk-meta", `reformulated to: ${entry.reformulated.query}`));
        }
        const list = el("div", "list");
        for (const hit of hits) {
          const support = supports[hit.chunk_id];
          const scoreBits = [];
          if (hit.rank !== undefined && hit.rank !== null) scoreBits.push(`rank ${fmt(hit.rank)}`);
          if (hit.dense_score !== undefined && hit.dense_score !== null) {
            scoreBits.push(`dense ${fmt(hit.dense_score)}`);
          }
          const rerankVal = (hit.rerank_score !== undefined && hit.rerank_score !== null)
            ? hit.rerank_score : hit.raw_score;
          if (rerankVal !== undefined && rerankVal !== null) scoreBits.push(`rerank ${fmt(rerankVal)}`);
          if (support !== undefined) scoreBits.push(`support ${fmt(support)}`);
          const scoreLabel = scoreBits.join(" | ");
          const chunkDetails = renderChunkDetails(row, hit.chunk_id);
          const summaryNode = chunkDetails.querySelector("summary");
          if (summaryNode) summaryNode.textContent = `${hit.chunk_id} (${scoreLabel})`;
          list.append(chunkDetails);
        }
        body.append(list);
        details.append(body);
        wrap.append(details);
      }
      return wrap;
    }

    function renderSlotCheck(row) {
      const section = el("section", "");
      const detail = row.plan_slot_detail;
      if (!Array.isArray(detail) || !detail.length) return section;
      section.append(el("h3", "", "Gold Evidence Slots vs Retrieval (plan_eval)"));
      const list = el("div", "list");
      for (const slot of detail) {
        const item = el("div", "list-row");
        const mark = el("span", slot.retrieved ? "slot-ok" : "slot-miss", slot.retrieved ? "RETRIEVED" : "MISSED");
        const meta = el("div", "meta");
        meta.append(mark);
        meta.append(document.createTextNode(
          `  role: ${slot.role} | gold: ${(slot.gold_chunk_ids || []).join(", ") || "none"}` +
          ` | surfaced by nodes: ${(slot.retrieved_by_nodes || []).join(", ") || "none"}`));
        item.append(meta);
        list.append(item);
      }
      section.append(list);
      return section;
    }

    function renderSelection(execution) {
      const wrap = el("div", "list");
      const selection = execution.selection || {};
      const header = chipRow([
        ["selection", execution.selection_policy || "n/a"],
        ["selectable candidates", selection.num_selectable],
        ["per-node keep", selection.per_node_keep],
        ["empty reason", execution.empty_reason],
      ]);
      wrap.append(header);
      const steps = selection.selection_steps || [];
      if (steps.length) {
        const details = el("details", "");
        details.append(el("summary", "", `Greedy selection steps (${steps.length})`));
        const body = el("div", "details-body");
        const list = el("div", "list");
        steps.forEach((step, index) => {
          const item = el("div", "list-row");
          item.append(el("div", "meta",
            `${index + 1}. ${step.unit_id} | gain ${fmt(step.marginal_gain)} ` +
            `(coverage ${fmt(step.coverage_gain)}, redundancy -${fmt(step.redundancy_penalty)}) ` +
            `| ${fmt(step.token_count)} tok | gain/tok ${fmt(step.gain_per_token)}`));
          const supports = step.selected_support_scores || {};
          item.append(el("div", "mono chunk-meta",
            "node support: " + Object.entries(supports).map(([n, s]) => `${n}=${fmt(s)}`).join("  ")));
          list.append(item);
        });
        body.append(list);
        details.append(body);
        wrap.append(details);
      }
      if (selection.final_coverage) {
        wrap.append(el("div", "mono chunk-meta",
          "final node coverage: " +
          Object.entries(selection.final_coverage).map(([n, c]) => `${n}=${fmt(c)}`).join("  ")));
      }
      if (selection.selected_unit_ids) {
        wrap.append(el("div", "mono chunk-meta",
          `top-per-node kept: ${selection.selected_unit_ids.join(", ") || "none"}`));
      }
      return wrap;
    }

    function renderFinalContext(row, execution) {
      const wrap = el("div", "list");
      const ordered = execution.final_ordering || [];
      if (!ordered.length) {
        wrap.append(el("div", "empty",
          `EMPTY CONTEXT — no candidate survived selection${execution.empty_reason ? ` (${execution.empty_reason})` : ""}.`));
        return wrap;
      }
      wrap.append(el("div", "chunk-meta",
        `${ordered.length} chunks | ${fmt(execution.final_token_count)} tokens of budget ${fmt(row.context_budget)}`));
      for (const chunkId of ordered) {
        wrap.append(renderChunkDetails(row, chunkId));
      }
      return wrap;
    }

    function renderAssembly(row) {
      const section = el("section", "");
      const trace = row.context_assembly_trace;
      if (!trace || typeof trace !== "object") return section;
      const method = trace.method || "";
      const isPlan = method === "plan_fixed" || method === "plan_dynamic";

      if (!isPlan) {
        section.append(el("h3", "", "Context Assembly Trace"));
        section.append(renderCollapsedPre(`method: ${method || "unknown"} (raw trace)`,
          JSON.stringify(trace, null, 2)));
        return section;
      }

      section.append(el("h3", "", "Plan & Evidence Assembly"));
      const stack = el("div", "list");

      // --- Stage 1: the plan --------------------------------------------------
      let nodes = [];
      if (method === "plan_fixed") {
        const decomposition = trace.decomposition || {};
        nodes = decomposition.nodes || [];
        stack.append(chipRow([
          ["pipeline", "fixed"],
          ["decomposition prompt", friendlyPrompt(decomposition.prompt_variant)],
          ["cache hit", decomposition.cache_hit],
          ["parse error", decomposition.parse_error],
          ["fallback node", decomposition.fallback],
        ]));
        if (decomposition.strategy) {
          stack.append(renderCollapsedPre("Model strategy (before nodes)", decomposition.strategy));
        }
        stack.append(el("h3", "", `Evidence Nodes (${nodes.length})`));
        stack.append(renderPlanNodes(nodes));
        stack.append(renderCollapsedPre("Decomposition prompt & raw output",
          `--- PROMPT ---\n${text(decomposition.prompt)}\n\n--- RAW OUTPUT ---\n${text(decomposition.raw_output)}`));
      } else {
        const contract = trace.contract || {};
        nodes = contract.nodes || [];
        stack.append(chipRow([
          ["pipeline", "dynamic"],
          ["planner prompt", trace.planner_prompt_variant],
          ["question type", contract.question_type],
          ["cache hit", trace.planner_cache_hit],
          ["contract deviations", (contract.deviations || []).length || "0"],
        ]));
        if (contract.strategy) {
          stack.append(renderCollapsedPre("Planner strategy", contract.strategy));
        }
        if ((contract.deviations || []).length) {
          stack.append(el("div", "mono chunk-meta", `deviations: ${(contract.deviations || []).join(" | ")}`));
        }
        stack.append(el("h3", "", `Evidence Nodes (${nodes.length})`));
        stack.append(renderPlanNodes(nodes));
        stack.append(renderCollapsedPre("Planner prompt & raw output",
          `--- PROMPT ---\n${text(trace.planner_prompt)}\n\n--- RAW OUTPUT ---\n${text(trace.planner_raw_output)}`));
      }

      // --- Executed policies ---------------------------------------------------
      const policies = trace.policies || {};
      stack.append(el("h3", "", "Executed Policies"));
      stack.append(chipRow([
        ["retrieval", policies.retrieval_mode],
        ["support", policies.support_policy],
        ["selection", policies.selection_policy],
        ["ordering", policies.ordering_policy],
      ]));

      const execution = trace.execution || {};

      // --- Stage 2: per-node retrieval -----------------------------------------
      stack.append(el("h3", "", "Per-Node Retrieval (with support scores)"));
      stack.append(renderPerNodeRetrieval(row, execution, nodes));

      // --- Gold slot check (from plan_eval sidecar) -----------------------------
      const slotSection = renderSlotCheck(row);
      if (slotSection.childNodes.length) stack.append(slotSection);

      // --- Stage 3: selection ---------------------------------------------------
      stack.append(el("h3", "", "Selection"));
      stack.append(renderSelection(execution));

      // --- Final context ----------------------------------------------------------
      stack.append(el("h3", "", "Final Context (after selection & ordering)"));
      stack.append(renderFinalContext(row, execution));

      section.append(stack);
      return section;
    }

    function renderResultCard(row) {
      const card = el("article", "result-card");
      const head = el("div", "result-head");
      head.append(el("div", "treatment-name", row.treatment || "unknown_treatment"));
      const metrics = el("div", "metrics");
      if (row.judge_rating !== undefined && row.judge_rating !== null) {
        metrics.append(badge("judge", `${fmt(row.judge_rating)}/5`, `judge ${judgeKind(row.judge_rating)}`));
      }
      metrics.append(badge("quote", row.evidence_quote_recall, recallKind(row.evidence_quote_recall)));
      metrics.append(badge("chunk", row.evidence_chunk_recall, recallKind(row.evidence_chunk_recall)));
      metrics.append(badge("ctx", row.context_tokens));
      metrics.append(badge("prompt", row.prompt_tokens));
      if (row.plan_slot_retrieval_recall !== undefined && row.plan_slot_retrieval_recall !== null) {
        metrics.append(badge("slotR", row.plan_slot_retrieval_recall, recallKind(row.plan_slot_retrieval_recall)));
      }
      if (row.evidence_role_recall !== undefined && row.evidence_role_recall !== null) {
        metrics.append(badge("roleR", row.evidence_role_recall, recallKind(row.evidence_role_recall)));
      }
      if (row.sufficient_context !== undefined && row.sufficient_context !== null) {
        metrics.append(badge("suff", row.sufficient_context, row.sufficient_context === 1 ? "good" : "bad"));
      }
      if (row.ci_positive_fraction !== undefined && row.ci_positive_fraction !== null) {
        metrics.append(badge("ci+", row.ci_positive_fraction, recallKind(row.ci_positive_fraction)));
      }
      head.append(metrics);
      card.append(head);

      const body = el("div", "result-body");
      body.append(renderPre("Model Output", row.model_output));
      body.append(renderJudge(row));
      body.append(renderAssembly(row));
      body.append(renderQuotes(row));
      body.append(renderEvidenceChunks(row));
      body.append(renderMetricAnnotations(row));
      body.append(renderIdLists(row));
      body.append(renderRetrieval(row));
      body.append(renderChunks(row));
      body.append(renderPrompts(row));
      card.append(body);
      return card;
    }

    function renderMetricAnnotations(row) {
      const section = el("section", "metric-annotations");
      const hasCi = Array.isArray(row.ci_values) && row.ci_values.length;
      const hasSuff = row.sufficient_context !== undefined && row.sufficient_context !== null;
      if (!hasCi && !hasSuff && !row.sufficient_context_explanation) return section;

      if (hasCi) {
        const details = el("details", "");
        details.append(el("summary", "", `CI values (base loss ${fmt(row.ci_base_loss)})`));
        const body = el("div", "details-body");
        const list = el("ul", "");
        row.ci_values.forEach((item) => {
          const phi = numberValue(item.phi);
          const marker = phi !== null && phi > 0 ? "+" : "-";
          list.append(el(
            "li",
            phi !== null && phi > 0 ? "ci-positive" : "ci-negative",
            `[${marker}] ${item.chunk_id}: phi=${fmt(item.phi)} (loss without: ${fmt(item.loss_without)})`,
          ));
        });
        body.append(list);
        details.append(body);
        section.append(details);
      }
      if (hasSuff || row.sufficient_context_explanation) {
        const details = el("details", "");
        details.append(el(
          "summary",
          "",
          `Sufficient context: ${row.sufficient_context === null || row.sufficient_context === undefined ? "n/a" : row.sufficient_context}`,
        ));
        const body = el("div", "details-body");
        body.append(el("pre", "", text(row.sufficient_context_explanation) || "no explanation recorded"));
        details.append(body);
        section.append(details);
      }
      return section;
    }

    function renderQuestionCard(questionId, groupRows) {
      const first = groupRows[0] || {};
      const card = el("article", "panel question-card");
      card.id = `q-${cssSafe(questionId)}`;
      const head = el("div", "question-head");
      head.append(el("div", "qid", questionId));
      head.append(el("h2", "", first.question || "No question text."));
      const expected = el("p", "expected", first.expected_answer || "No expected answer recorded.");
      head.append(expected);
      card.append(head);
      const treatments = el("div", "treatments");
      groupRows.forEach((row) => treatments.append(renderResultCard(row)));
      card.append(treatments);
      return card;
    }

    function populateFilters() {
      const treatments = ["all", ...unique(rows.map((row) => row.treatment)).sort()];
      const questions = ["all", ...unique(rows.map((row) => row.question_id))];
      const previousTreatment = els.treatmentSelect.value || "all";
      const previousQuestion = els.questionSelect.value || "all";

      els.treatmentSelect.replaceChildren(...treatments.map((value) => el("option", "", value)));
      els.questionSelect.replaceChildren(...questions.map((value) => {
        const option = el("option", "", value);
        option.value = value;
        return option;
      }));

      els.treatmentSelect.value = treatments.includes(previousTreatment) ? previousTreatment : "all";
      els.questionSelect.value = questions.includes(previousQuestion) ? previousQuestion : "all";
    }

    function render() {
      const activeRows = filteredRows();
      renderStats(activeRows);
      renderSummary(activeRows);
      renderNav(activeRows);

      const groups = groupByQuestion(activeRows);
      const cards = [];
      for (const [questionId, groupRows] of groups.entries()) {
        cards.push(renderQuestionCard(questionId, groupRows));
      }
      if (!cards.length) {
        els.results.replaceChildren(el("div", "panel empty", "No matching rows."));
      } else {
        els.results.replaceChildren(...cards);
      }
    }

    function loadRows(newRows, source) {
      rows = newRows;
      selectedQuestion = "all";
      currentResultSource = source;
      updateSourceLabel();
      populateFilters();
      render();
    }

    els.searchInput.addEventListener("input", render);
    els.treatmentSelect.addEventListener("change", render);
    els.questionSelect.addEventListener("change", () => {
      selectedQuestion = els.questionSelect.value;
      render();
    });
    els.promptToggle.addEventListener("change", render);
    els.resultsFileInput.addEventListener("change", () => {
      const file = els.resultsFileInput.files && els.resultsFileInput.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {
        try {
          loadRows(parseRows(String(reader.result || "")), file.name);
        } catch (error) {
          els.results.replaceChildren(el("div", "panel empty", error.message));
        }
      };
      reader.readAsText(file);
    });
    els.chunksFileInput.addEventListener("change", () => {
      const file = els.chunksFileInput.files && els.chunksFileInput.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {
        try {
          chunksById = parseChunks(String(reader.result || ""));
          currentChunkSource = file.name;
          updateSourceLabel();
          render();
        } catch (error) {
          els.results.replaceChildren(el("div", "panel empty", error.message));
        }
      };
      reader.readAsText(file);
    });

    loadEmbeddedData();
    populateFilters();
    render();
  </script>
</body>
</html>
"""


def _json_for_script(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False).replace("</", "<\\/")


def render_results_html(
    rows: list[dict[str, Any]],
    source_path: str | Path,
    chunk_lookup: dict[str, dict[str, Any]] | None = None,
    chunk_source_path: str | Path | None = None,
) -> str:
    source_label = str(source_path)
    chunk_source_label = "" if chunk_source_path is None else str(chunk_source_path)
    return (
        HTML_TEMPLATE.replace("__RESULT_DATA__", _json_for_script(rows))
        .replace("__CHUNK_DATA__", _json_for_script(chunk_lookup or {}))
        .replace("__SOURCE_LABEL__", json.dumps(source_label))
        .replace("__CHUNK_SOURCE_LABEL__", json.dumps(chunk_source_label))
    )


def chunk_lookup_from_rows(chunks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(chunk["chunk_id"]): dict(chunk)
        for chunk in chunks
        if isinstance(chunk, dict) and "chunk_id" in chunk
    }


def infer_chunks_path(
    results_path: str | Path,
    rows: list[dict[str, Any]],
    chunks_path: str | Path | None = None,
) -> Path | None:
    if chunks_path is not None:
        candidate = Path(chunks_path)
        if not candidate.exists():
            raise FileNotFoundError(f"Chunk file not found: {candidate}")
        return candidate

    result_path = Path(results_path)
    candidates: list[Path] = [result_path.parent / "hamlet_chunks.jsonl"]
    for row in rows:
        run_config = row.get("run_config")
        if isinstance(run_config, dict) and run_config.get("chunks_path"):
            candidates.append(Path(str(run_config["chunks_path"])))
    candidates.append(Path("data/hamlet_chunks.jsonl"))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def write_results_html(
    results_path: str | Path,
    output_path: str | Path,
    chunks_path: str | Path | None = None,
) -> Path:
    from hamlet_qa.metrics.annotate import (
        load_annotations,
        merge_annotations_into_rows,
    )

    rows = load_result_rows(results_path)
    annotations = load_annotations(results_path)
    if annotations:
        rows = merge_annotations_into_rows(rows, annotations)
    resolved_chunks_path = infer_chunks_path(results_path, rows, chunks_path)
    chunk_lookup: dict[str, dict[str, Any]] = {}
    if resolved_chunks_path is not None:
        chunk_lookup = chunk_lookup_from_rows(load_jsonl(resolved_chunks_path))
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        render_results_html(rows, results_path, chunk_lookup, resolved_chunks_path),
        encoding="utf-8",
    )
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Hamlet QA result JSON/JSONL as a static HTML viewer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("results", help="Path to results.jsonl or a JSON file.")
    parser.add_argument(
        "--output",
        "-o",
        default="results_viewer.html",
        help="HTML output path.",
    )
    parser.add_argument(
        "--chunks",
        help=(
            "Chunk JSONL/JSON file to embed for evidence chunk expansion. "
            "Defaults to hamlet_chunks.jsonl beside the results file, then the "
            "run_config chunks_path, then data/hamlet_chunks.jsonl."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = write_results_html(args.results, args.output, args.chunks)
    print(f"Wrote HTML results viewer to {output_path}")


if __name__ == "__main__":
    main()
