"""Live monitoring dashboard for Claude Codex.

Serves a self-contained HTML page at /dashboard and a JSON feed at
/dashboard/data, both reading the per-request files in logs/requests_full/.
No external assets, no build step - view it, hack it.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from claudecodex.logging_config import _FULL_DIR


def _summarize(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce one full request record to what the dashboard table shows."""
    response = entry.get("response") or {}
    usage = response.get("usage") or {}

    tools_used = []
    preview = ""
    for block in response.get("content") or []:
        if block.get("type") == "tool_use":
            tools_used.append(block.get("name", "?"))
        elif block.get("type") == "text" and not preview:
            preview = block.get("text", "")[:100]

    return {
        "id": entry.get("id"),
        "timestamp": entry.get("timestamp"),
        "endpoint": entry.get("endpoint"),
        "provider": entry.get("provider"),
        "model": entry.get("model"),
        "status_code": entry.get("status_code"),
        "duration_ms": entry.get("duration_ms"),
        "input_tokens": usage.get("input_tokens", response.get("input_tokens", 0)),
        "output_tokens": usage.get("output_tokens", 0),
        "stop_reason": response.get("stop_reason"),
        "streamed": bool(response.get("streamed")),
        "tools_used": tools_used,
        "preview": preview,
        "error": (entry.get("error") or "")[:200] or None,
    }


def read_recent_requests(limit: int = 200) -> List[Dict[str, Any]]:
    """Read the newest request records, most recent first."""
    log_dir = Path(_FULL_DIR)
    if not log_dir.is_dir():
        return []

    files = sorted(
        log_dir.glob("*.json"), key=os.path.getmtime, reverse=True
    )[:limit]

    records = []
    for path in files:
        try:
            with open(path) as f:
                records.append(_summarize(json.load(f)))
        except (json.JSONDecodeError, OSError):
            continue
    return records


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Claude Codex — Live Monitor</title>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --ink: #e6edf3; --ink-2: #9198a1; --ink-3: #6e7681;
    --good: #3fb950; --bad: #f85149; --accent: #58a6ff; --warn: #d29922;
  }
  * { box-sizing: border-box; margin: 0; }
  body {
    background: var(--bg); color: var(--ink);
    font: 14px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace;
    padding: 24px; max-width: 1200px; margin: 0 auto;
  }
  header { display: flex; align-items: baseline; gap: 12px; margin-bottom: 20px; }
  h1 { font-size: 18px; font-weight: 600; }
  #meta { color: var(--ink-3); font-size: 12px; }
  .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
           gap: 12px; margin-bottom: 20px; }
  .tile { background: var(--surface); border: 1px solid var(--border);
          border-radius: 8px; padding: 12px 16px; }
  .tile .label { color: var(--ink-2); font-size: 11px; text-transform: uppercase;
                 letter-spacing: .06em; }
  .tile .value { font-size: 24px; font-weight: 600; margin-top: 2px; }
  .tile .sub { color: var(--ink-3); font-size: 11px; }
  .controls { display: flex; gap: 8px; margin-bottom: 12px; }
  .controls button {
    background: var(--surface); color: var(--ink-2); border: 1px solid var(--border);
    border-radius: 6px; padding: 4px 12px; font: inherit; font-size: 12px; cursor: pointer;
  }
  .controls button.active { color: var(--ink); border-color: var(--accent); }
  .table-wrap { overflow-x: auto; border: 1px solid var(--border);
                border-radius: 8px; background: var(--surface); }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th, td { padding: 7px 10px; text-align: left; border-bottom: 1px solid var(--border);
           white-space: nowrap; }
  th { color: var(--ink-2); font-weight: 500; font-size: 11px;
       text-transform: uppercase; letter-spacing: .05em; }
  td.preview { white-space: normal; color: var(--ink-2); max-width: 320px; }
  .ok { color: var(--good); } .err { color: var(--bad); }
  .badge { border: 1px solid var(--border); border-radius: 10px; padding: 1px 8px;
           font-size: 11px; color: var(--ink-2); }
  .tool { color: var(--warn); }
  .stream { color: var(--accent); }
  #empty { color: var(--ink-3); padding: 32px; text-align: center; }
</style>
</head>
<body>
<header>
  <h1>Claude Codex — Live Monitor</h1>
  <span id="meta">loading…</span>
</header>

<div class="tiles" id="tiles"></div>

<div class="controls" id="controls">
  <button data-f="all" class="active">All</button>
  <button data-f="messages">Messages</button>
  <button data-f="tools">Tool calls</button>
  <button data-f="errors">Errors</button>
</div>

<div class="table-wrap">
<table>
  <thead><tr>
    <th>Time</th><th>Endpoint</th><th>Model</th><th>Status</th><th>ms</th>
    <th>Tokens in→out</th><th>Stop</th><th>Tools</th><th>Response preview</th>
  </tr></thead>
  <tbody id="rows"></tbody>
</table>
</div>
<div id="empty" hidden>No requests logged yet — point Claude Code at this proxy.</div>

<script>
let filter = "all", data = [];

document.getElementById("controls").addEventListener("click", e => {
  if (!e.target.dataset.f) return;
  filter = e.target.dataset.f;
  document.querySelectorAll(".controls button").forEach(b =>
    b.classList.toggle("active", b.dataset.f === filter));
  render();
});

function esc(s) {
  return String(s ?? "").replace(/[&<>"]/g,
    c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
}

function fmtTime(ts) {
  const d = new Date(ts);
  return isNaN(d) ? esc(ts) : d.toLocaleTimeString();
}

function render() {
  const rows = data.filter(r =>
    filter === "all" ? true :
    filter === "errors" ? r.status_code >= 400 :
    filter === "tools" ? r.tools_used.length > 0 :
    r.endpoint === "/v1/messages");

  const total = data.length;
  const errors = data.filter(r => r.status_code >= 400).length;
  const msgs = data.filter(r => r.endpoint === "/v1/messages" && r.status_code < 400);
  const avgMs = msgs.length
    ? Math.round(msgs.reduce((a, r) => a + (r.duration_ms || 0), 0) / msgs.length) : 0;
  const tin = data.reduce((a, r) => a + (r.input_tokens || 0), 0);
  const tout = data.reduce((a, r) => a + (r.output_tokens || 0), 0);
  const models = [...new Set(data.map(r => r.model).filter(Boolean))];

  document.getElementById("tiles").innerHTML = `
    <div class="tile"><div class="label">Requests</div>
      <div class="value">${total}</div><div class="sub">last ${total} on disk</div></div>
    <div class="tile"><div class="label">Errors</div>
      <div class="value ${errors ? "err" : "ok"}">${errors}</div>
      <div class="sub">${total ? (100 * errors / total).toFixed(1) : 0}% of requests</div></div>
    <div class="tile"><div class="label">Avg latency</div>
      <div class="value">${avgMs}<span style="font-size:13px"> ms</span></div>
      <div class="sub">completions only</div></div>
    <div class="tile"><div class="label">Tokens</div>
      <div class="value">${(tin + tout).toLocaleString()}</div>
      <div class="sub">${tin.toLocaleString()} in · ${tout.toLocaleString()} out</div></div>
    <div class="tile"><div class="label">Models</div>
      <div class="value">${models.length}</div>
      <div class="sub">${esc(models.slice(0, 2).join(", "))}</div></div>`;

  document.getElementById("rows").innerHTML = rows.map(r => `
    <tr>
      <td>${fmtTime(r.timestamp)}</td>
      <td><span class="badge">${esc((r.endpoint || "").replace("/v1/", ""))}</span>
          ${r.streamed ? '<span class="stream">stream</span>' : ""}</td>
      <td>${esc(r.model)}</td>
      <td class="${r.status_code >= 400 ? "err" : "ok"}">${r.status_code}
          ${r.status_code >= 400 ? "✗" : "✓"}</td>
      <td>${r.duration_ms == null ? "" : Math.round(r.duration_ms)}</td>
      <td>${r.input_tokens || 0} → ${r.output_tokens || 0}</td>
      <td>${esc(r.stop_reason || "")}</td>
      <td class="tool">${esc(r.tools_used.join(", "))}</td>
      <td class="preview">${r.error
        ? `<span class="err">${esc(r.error)}</span>` : esc(r.preview)}</td>
    </tr>`).join("");

  document.getElementById("empty").hidden = rows.length > 0;
  document.getElementById("meta").textContent =
    `auto-refresh 3s · ${new Date().toLocaleTimeString()}`;
}

async function refresh() {
  try {
    data = await (await fetch("/dashboard/data")).json();
    render();
  } catch (e) {
    document.getElementById("meta").textContent = "feed unavailable: " + e;
  }
}

refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>
"""
