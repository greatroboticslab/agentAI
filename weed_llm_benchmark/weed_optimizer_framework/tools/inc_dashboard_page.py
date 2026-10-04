"""The /inc page (served by inc_dashboard.py), kept apart from the routes.

The page is a Python string, so the script in it is written without a single
backslash (Python would eat it before the browser saw it) and without inline
event handlers: every control carries data attributes and one delegated
listener acts on them (the same rules brain/api.py's supervision page follows;
tests/test_inc_ap_dashboard.py checks the served script).
"""

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>INC campaign</title>
<style>
 body{margin:0;padding:1rem;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
 h1{margin:.2rem 0 .1rem}
 h4{margin:.8rem 0 .3rem;font-size:.9rem}
 .sub{opacity:.7;font-size:.85rem;margin-bottom:.6rem}
 .tiny{opacity:.75;font-size:.75rem}
 .card{border:1px solid rgba(255,255,255,.12);border-radius:10px;padding:.9rem 1rem;
       margin:0;background:rgba(255,255,255,.03);min-width:0}
 .card h3{margin:0 0 .5rem;font-size:1rem}
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,340px),1fr));gap:1rem;
       align-items:start}
 .wide{grid-column:1/-1}
 .two{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:1rem}
 table{width:100%;border-collapse:collapse;font-size:.83rem}
 td,th{text-align:left;padding:.28rem .45rem;border-bottom:1px solid rgba(255,255,255,.08);
       vertical-align:top;overflow-wrap:break-word}
 th{opacity:.65;font-weight:600}
 .scroll{overflow-x:auto;max-height:28rem;overflow-y:auto}
 .scroll th{position:sticky;top:0;background:#0e1626}
 code{font-size:.78rem;word-break:break-all}
 .pill{display:inline-block;padding:.05rem .5rem;border-radius:999px;font-size:.72rem;
       border:1px solid currentColor}
 .crit{color:#ff6b6b}.warn{color:#ffb454}.info{color:#7fb3ff}.ok{color:#5fd08a}
 .missing{opacity:.6;font-style:italic}
 .err{color:#ff6b6b;font-size:.8rem;min-height:1em}
 .okmsg{color:#5fd08a;font-size:.8rem;min-height:1em}
 .v-ACCEPT{color:#5fd08a;font-weight:600}.v-HOLD{color:#ffb454;font-weight:600}
 .v-REJECT{color:#ff6b6b;font-weight:600}
 .bar{height:10px;border-radius:6px;background:rgba(255,255,255,.08);overflow:hidden;display:flex;
      margin:.3rem 0}
 .bar span{display:block;height:100%}
 .spent{background:#5fd08a}.committed{background:#ffb454}
 .ctl{display:flex;flex-wrap:wrap;gap:.4rem;align-items:center;margin:.35rem 0}
 .ctl label{font-size:.8rem;opacity:.75}
 input.txt,select{font:inherit;padding:.15rem .4rem;border-radius:6px;min-width:6rem;flex:1 1 8rem;
       max-width:100%}
 input.num{font:inherit;padding:.15rem .4rem;border-radius:6px;width:6rem}
 button{font:inherit;padding:.15rem .6rem;border-radius:6px;cursor:pointer}
 .item{border-top:1px solid rgba(255,255,255,.08);padding:.45rem 0}
 .diag{border-top:1px solid rgba(255,255,255,.08);padding:.5rem 0}
 ul.cites{margin:.2rem 0 0;padding-left:1.1rem;font-size:.8rem}
 ul.tree{list-style:none;padding-left:1rem;margin:.2rem 0;border-left:1px solid rgba(255,255,255,.12)}
 ul.tree li{margin:.3rem 0}
 .edge{font-size:.75rem;opacity:.8}
 .disp{border:1px dashed #ffb454;padding:.5rem .7rem;border-radius:8px;margin-bottom:.5rem;
       font-size:.85rem}
 pre.led{white-space:pre-wrap;word-break:break-word;font-size:.72rem;max-height:16rem;overflow:auto;
         margin:.2rem 0 .6rem}
 .hl{outline:1px solid #7fb3ff;border-radius:6px;padding:.2rem}
 details summary{cursor:pointer;font-size:.82rem;opacity:.85;margin:.3rem 0}
</style></head><body>
<h1>INC campaign</h1>
<div class="sub">The platform builds, advances, diagnoses and follows up incremental-training (INC)
experiments; the per-increment accept or reject stays with the pinned gate. Every value on this page is
read from what the campaign ticker recorded on the lab (its state, snapshots, ledger copies, diagnoses) and
the approval queue: opening it does not reach the cluster. A button does, once, through the policy gate and
the executor, in your name.
<a href="/supervision/weed">Supervision</a> &middot; <a href="/agent/weed">Project</a> &middot;
<a href="#" data-refresh="1">Refresh</a></div>
<div id="msg" class="okmsg"></div>
<div class="grid">
 <div class="card wide"><h3>Campaign</h3><div id="b_campaign">loading&hellip;</div></div>
 <div class="card wide"><h3>Lineage</h3><div id="b_lineage">loading&hellip;</div></div>
 <div class="card wide"><h3>Current experiment</h3><div id="b_exp">loading&hellip;</div></div>
 <div class="card wide" id="c_ledger" style="display:none"><h3>Ledger</h3><div id="b_ledger"></div></div>
 <div class="card wide"><h3>Diagnoses</h3><div id="b_diag">loading&hellip;</div></div>
 <div class="card wide"><h3>Proposals</h3><div id="b_props">loading&hellip;</div></div>
 <div class="card"><h3>Human research cards (R4)</h3><div id="b_cards">loading&hellip;</div></div>
 <div class="card"><h3>Track record</h3><div id="b_track">loading&hellip;</div></div>
 <div class="card"><h3>Replay and prospective tests</h3><div id="b_replay">loading&hellip;</div></div>
 <div class="card wide"><h3>Final table of the report (display only)</h3><div id="b_final">loading&hellip;</div></div>
</div>
<script>
const Q = new URLSearchParams(location.search);
const S = {campaign: Q.get("campaign") || "", exp: Q.get("exp") || "", current: null, known: []};
const DOMAIN = "weed";
function esc(s){return String(s==null?"":s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));}
function short(s, n){ s = String(s==null?"":s); return s.length > n ? s.slice(0, n) + "..." : s; }
function pill(t, cls){ return '<span class="pill '+esc(cls||t)+'">'+esc(t)+'</span>'; }
function num(v, d){
  if(v==null || v==="") return "-";
  const x = Number(v);
  return isFinite(x) ? x.toFixed(d==null?3:d) : esc(v);
}
function ago(s){
  if(s==null) return "never";
  s = Math.max(0, Math.round(s));
  if(s < 120) return s + " s ago";
  if(s < 7200) return Math.round(s/60) + " min ago";
  if(s < 172800) return Math.round(s/3600) + " h ago";
  return Math.round(s/86400) + " d ago";
}
function table(head, rows){
  if(!rows.length) return '<div class="missing">nothing recorded yet</div>';
  return '<div class="scroll"><table><tr>'+head.map(h=>'<th>'+esc(h)+'</th>').join('')+'</tr>'
    + rows.map(r=>'<tr>'+r.map(c=>'<td>'+c+'</td>').join('')+'</tr>').join('') + '</table></div>';
}
function kv(pairs){ return table(["", ""], pairs.filter(p => p).map(p => ['<span class="tiny">'+esc(p[0])+'</span>', p[1]])); }
function unavailable(d){ return '<div class="missing">Not available &mdash; '+esc((d && d.reason) || "no data")+'</div>'; }
function fail(e){ return '<div class="crit">could not load: '+esc(e)+'</div>'; }
function set(id, html){ const el = document.getElementById(id); if(el) el.innerHTML = html; }
function val(id){ const el = document.getElementById(id); return el ? String(el.value || "").trim() : ""; }
function say(t, bad){ const m = document.getElementById("msg"); m.className = bad ? "err" : "okmsg"; m.textContent = t; }
function q(extra){
  const p = new URLSearchParams();
  if(S.campaign) p.set("campaign", S.campaign);
  const x = extra || {};
  Object.keys(x).forEach(k => { if(x[k] != null && x[k] !== "") p.set(k, x[k]); });
  const s = p.toString();
  return s ? "?" + s : "";
}
function getJSON(url){ return fetch(url, {credentials: "same-origin"}).then(r => r.json()); }
function postJSON(url, body){
  return fetch(url, {method: "POST", credentials: "same-origin",
                     headers: {"Content-Type": "application/json"}, body: JSON.stringify(body || {})})
    .then(r => r.json().catch(() => ({})).then(d => { d._status = r.status; return d; }));
}
function citeHtml(c){
  if(!c) return "";
  const addr = esc(c.artifact) + (c.line ? ":" + esc(c.line) : "") + (c.pointer ? "#" + esc(c.pointer) : "");
  const v = '<code>' + esc(short(JSON.stringify(c.value), 140)) + '</code>';
  if(c.ledger_url) return '<a href="#" data-ledger="'+esc(c.ledger_url)+'"><code>'+addr+'</code></a> = ' + v;
  return '<code>' + addr + '</code> = ' + v;
}
function burn(b){
  if(!b || b.error) return '<span class="missing">budget unavailable' + (b && b.error ? ': ' + esc(b.error) : '') + '</span>';
  const env = Number(b.envelope_su) || 0, sp = Number(b.spent_su) || 0, cm = Number(b.committed_su) || 0;
  const w1 = env ? Math.min(100, 100 * sp / env) : 0, w2 = env ? Math.min(100 - w1, 100 * cm / env) : 0;
  return '<div class="bar" title="spent (green) and committed (amber) of the envelope">'
    + '<span class="spent" style="width:' + w1.toFixed(1) + '%"></span>'
    + '<span class="committed" style="width:' + w2.toFixed(1) + '%"></span></div>'
    + '<div class="tiny">' + num(sp, 1) + ' SU spent + ' + num(cm, 1) + ' SU committed of ' + num(env, 0)
    + ' SU; ' + num(b.remaining_su, 1) + ' SU left. Today ' + num(b.today_su, 1)
    + (b.daily_cap_su == null ? ' SU (no daily cap).' : ' of ' + num(b.daily_cap_su, 0) + ' SU.') + '</div>';
}
function healthLine(h){
  if(!h) return "";
  const cls = h.level === "ok" ? "ok" : h.level === "warn" ? "warn" : "crit";
  return pill(h.level, cls) + ' <span class="tiny">' + esc(h.reason) + '</span>';
}
function goalText(g){
  if(g == null || g === "") return '<span class="missing">not set: the campaign runs until nothing is left to propose</span>';
  if(g.kind === "diagnosis") return 'diagnosis <b>' + esc(g.id) + ' ' + esc(g.name) + '</b> fires';
  if(g.kind === "exp_done") return 'experiment <b>' + esc(g.exp) + '</b> is done';
  return '<span class="warn">not a goal the ticker reads: ' + esc(JSON.stringify(g)) + '</span>';
}
function goalEditor(c, opts){
  const g = c.goal || {};
  const kind = g.kind || "none";
  const dval = g.kind === "diagnosis" ? g.id + ":" + g.name : "";
  const eval_ = g.kind === "exp_done" ? g.exp : "";
  return '<div class="ctl"><label>goal</label><select id="in_goal_kind">'
    + [["none", "no goal"], ["diagnosis", "a diagnosis fires"], ["exp_done", "an experiment is done"]].map(o =>
        '<option value="' + o[0] + '"' + (o[0] === kind ? ' selected' : '') + '>' + o[1] + '</option>').join('')
    + '</select><input class="txt" id="in_goal_diag" list="goal_opts" placeholder="ID:NAME, e.g. D4:decision_slot_ready" value="' + esc(dval) + '">'
    + '<input class="txt" id="in_goal_exp" placeholder="experiment, e.g. pilot_v2" value="' + esc(eval_) + '">'
    + '<button data-camp="goal">Set goal</button></div>'
    + '<datalist id="goal_opts">' + (opts || []).map(o => '<option value="' + esc(o.id + ":" + o.name) + '">').join('') + '</datalist>'
    + '<div class="tiny">When the goal is met at a diagnosis the campaign completes with a card. Goals are judged on dev only.</div>';
}
function createForm(){
  return '<h4>Start a campaign</h4><div class="ctl"><input class="txt" id="in_cname" placeholder="campaign name">'
    + '<input class="txt" id="in_cexp" placeholder="experiment it follows first, e.g. pilot_v2">'
    + '<button data-create="1">Create (disabled until enabled)</button></div>'
    + '<div class="tiny">Administrators only. A new campaign starts disabled with autonomy off.</div>';
}
function renderCampaign(d, c){
  if(!d.available) return unavailable(d);
  const cs = d.campaigns || [];
  if(!c) return '<div class="missing">' + esc(d.reason || "no campaign") + '</div>' + healthLine(d.health) + createForm();
  let h = '';
  if(cs.length > 1){
    h += '<div class="ctl"><label>campaign</label><select data-sel="campaign">'
      + cs.map(x => '<option' + (x.name === c.name ? ' selected' : '') + ' value="' + esc(x.name) + '">' + esc(x.name) + '</option>').join('')
      + '</select></div>';
  }
  const status = c.paused_reason ? pill("paused", "crit") : c.enabled ? pill("running", "ok") : pill("disabled", "info");
  const obs = c.observed || {};
  const every = Math.round((d.ticker_expected_s || 600) / 60);
  h += '<div><b>' + esc(c.name) + '</b> ' + status + ' ' + healthLine(d.health) + '</div>';
  if(d.tree_mismatch) h += '<div class="crit tiny">' + esc(d.tree_mismatch) + '</div>';
  if(c.paused_reason) h += '<div class="crit tiny">' + esc(c.paused_reason) + '</div>';
  if(c.card) h += '<div class="' + (c.card.kind === "paused" ? "crit" : "warn") + '"><b>' + esc(c.card.title) + '</b> <span class="tiny">' + esc(c.card.utc || "") + '</span><div class="tiny">' + esc(short(c.card.detail, 600)) + '</div></div>';
  const it = c.item;
  h += kv([
    ["goal", goalText(c.goal)],
    ["phase", c.phase ? esc(c.phase) : '<span class="missing">the ticker has not run this campaign yet</span>'],
    ["current experiment", c.exp ? '<a href="#" data-exp="' + esc(c.exp) + '">' + esc(c.exp) + '</a>'
      + ' <span class="tiny">' + esc([obs.built ? "built" : (obs.built === false ? "not built" : ""), obs.done ? "done" : "",
        obs.generation != null ? "generation " + obs.generation : "", obs.n_blocked ? obs.n_blocked + " blocked" : "",
        obs.abandoned ? "abandoned" : ""].filter(x => x).join(", ")) + '</span>'
      + (c.current_source && c.current_source !== "ticker state" ? ' <span class="tiny warn">(' + esc(c.current_source) + ')</span>' : '')
      : '<span class="missing">none</span>'],
    ["in flight", it ? '<b>' + esc(it.lever) + '</b> ' + esc(it.action) + ' <span class="tiny">' + esc(it.status) + (it.approval_id ? ' &middot; approval ' + esc(it.approval_id) : '') + '</span>'
      + ((it.argv || []).length ? '<div><code>' + esc(it.argv.join(" ")) + '</code></div>' : '')
      : (c.building ? 'building ' + esc(c.building.exp) + ' <span class="tiny">jobs ' + esc((c.building.job_ids || []).join(", ")) + '</span>'
         : (c.wait_jobs || []).length ? 'waiting for jobs ' + esc(c.wait_jobs.join(", ")) : '<span class="missing">nothing</span>')],
    ["last tick", c.last_tick_utc ? esc(c.last_tick_utc) + ' <span class="tiny">(' + ago(c.last_tick_age_s) + ', one every ' + every + ' min; tick ' + esc(c.ticks) + ')</span>' : '<span class="missing">never</span>'],
    ["last snapshot", c.last_snapshot_utc ? esc(c.last_snapshot_utc) + ' <span class="tiny">(' + ago(c.last_snapshot_age_s) + '; taken while the campaign runs or waits on a job)</span>'
      + (c.snapshot_failures ? ' <span class="warn tiny">' + esc(c.snapshot_failures) + ' failed in a row</span>' : '') : '<span class="missing">none yet</span>'],
    c.errors ? ["ticker errors", '<span class="warn">' + esc(c.errors) + ' in a row: ' + esc(short((c.last_error || {}).error, 300)) + '</span>'] : null,
    ["brain", c.brain ? esc("plan " + c.brain.n + " for " + c.brain.exp + ": " + c.brain.status) + (c.brain.reason ? ' <span class="tiny">' + esc(short(c.brain.reason, 200)) + '</span>' : '')
      : '<span class="missing">' + (c.brain_enabled ? 'no plan yet' : 'off for this campaign') + '</span>'],
    ["autonomy", esc(c.autonomy) + (c.autonomy === "envelope" ? ' <span class="tiny">granted by ' + esc(c.autonomy_granted_by) + '</span>' : ' <span class="tiny">R3 builds wait for a person</span>')],
    ["replay gate", (d.replay && d.replay.passed ? pill("passed", "ok") : pill("not passed", "warn")) + ' <span class="tiny">' + esc(d.replay && d.replay.reason) + '</span>'],
    ["budget", burn(c.budget)],
    ["heartbeat", d.heartbeat && d.heartbeat.utc ? esc(d.heartbeat.utc) + ' <span class="tiny">(' + ago(d.heartbeat.age_s) + ')</span>' : '<span class="missing">the ticker has written no status yet</span>']
  ]);
  const env = (c.budget && c.budget.envelope_su != null) ? c.budget.envelope_su : (c.envelope_su != null ? c.envelope_su : "");
  const daily = (c.budget && c.budget.daily_cap_su != null) ? c.budget.daily_cap_su : (c.daily_cap_su != null ? c.daily_cap_su : "");
  h += '<h4>Controls <span class="tiny">(administrators; every change goes to the campaign ledger with your name)</span></h4>';
  h += '<div class="ctl">' + (c.enabled ? '<button data-camp="disable">Disable</button>' : '<button data-camp="enable">Enable</button>')
    + (c.paused_reason ? '<button data-camp="resume">Resume</button>'
       : '<input class="txt" id="in_pause" placeholder="why pause"><button data-camp="pause">Pause</button>') + '</div>';
  h += goalEditor(c, d.goal_options);
  h += '<div class="ctl"><label>envelope SU</label><input class="num" id="in_env" type="number" min="0" value="' + esc(env) + '">'
    + '<label>daily cap SU</label><input class="num" id="in_daily" type="number" min="0" value="' + esc(daily) + '">'
    + '<button data-camp="budget">Set budget</button></div>';
  h += '<div class="ctl">' + (c.autonomy === "envelope" ? '<button data-camp="autonomy-off">Turn envelope autonomy off</button>'
       : '<button data-camp="autonomy-on">Turn envelope autonomy on</button>')
    + '<span class="tiny">When on, R3 builds of L1, L2, L5, L8 and L9 run inside the envelope without asking, once the replay tests pass on this code.</span></div>';
  h += '<div class="ctl"><input class="txt" id="in_kill" placeholder="why kill the current experiment"><button data-camp="kill">Pause and kill ' + esc(c.exp || "the current experiment") + '</button></div>';
  return h;
}
function lineageHtml(d){
  if(!d.available) return unavailable(d);
  if(!(d.nodes || []).length) return '<div class="missing">' + esc(d.reason || "no experiment yet") + '</div>';
  const byExp = {};
  d.nodes.forEach(n => { byExp[n.exp] = n; });
  const kids = {};
  (d.edges || []).forEach(e => { const p = e.parent || ""; (kids[p] = kids[p] || []).push(e); });
  const seen = {};
  function nodeLabel(x){
    const n = byExp[x] || {exp: x};
    const tags = [];
    if(n.builder) tags.push(n.builder);
    if(n.replay_mode) tags.push("replay " + n.replay_mode);
    if(n.done === true) tags.push("done"); else if(n.done === false) tags.push("running");
    else if(n.built) tags.push("built"); else if(n.built === false) tags.push("not built");
    if(n.abandoned || n.cancelled) tags.push("cancelled");
    return '<a href="#" data-exp="' + esc(x) + '"><b>' + esc(x) + '</b></a>' + (tags.length ? ' <span class="tiny">' + esc(tags.join(", ")) + '</span>' : '');
  }
  function edgeLabel(e){
    const bits = [];
    if(e.lever) bits.push(e.lever);
    if((e.trigger || []).length) bits.push("trigger " + e.trigger.join(", "));
    if((e.approval_ids || []).length) bits.push("approval " + e.approval_ids.join(", "));
    if(e.status) bits.push(e.status);
    if(e.decided_by) bits.push("by " + e.decided_by);
    if((e.job_ids || []).length) bits.push("job " + e.job_ids.join(", "));
    return '<span class="edge">' + esc(bits.join(" | ") || "no provenance recorded") + '</span>';
  }
  function sub(x){
    if(seen[x]) return "";
    seen[x] = 1;
    const es = kids[x] || [];
    if(!es.length) return "";
    return '<ul class="tree">' + es.map(e => '<li>' + edgeLabel(e) + '<br>&rarr; ' + nodeLabel(e.child) + sub(e.child) + '</li>').join('') + '</ul>';
  }
  let h = '<ul class="tree">' + (d.roots || []).map(r => '<li>' + nodeLabel(r) + sub(r) + '</li>').join('') + '</ul>';
  if(kids[""]) h += '<div class="tiny">built with no recorded parent</div><ul class="tree">'
    + kids[""].map(e => '<li>' + edgeLabel(e) + '<br>&rarr; ' + nodeLabel(e.child) + sub(e.child) + '</li>').join('') + '</ul>';
  return h + '<div class="tiny">An edge names the lever, the diagnoses that triggered it and the approval that let it run.</div>';
}
function expPicker(){
  const names = (S.known || []).slice();
  if(S.exp && names.indexOf(S.exp) < 0) names.push(S.exp);
  if(!names.length) return '';
  return '<div class="ctl"><label>experiment</label><select data-sel="exp">'
    + names.map(n => '<option' + (n === S.exp ? ' selected' : '') + ' value="' + esc(n) + '">' + esc(n) + (n === S.current ? ' (current)' : '') + '</option>').join('')
    + '</select></div>';
}
function gridHtml(g){
  if(!g || !(g.rows || []).length) return '<div class="missing">no decided step yet</div>';
  const head = ["step", "truth"].concat(g.chains);
  const rows = g.rows.map(r => {
    let step = esc(r.k) + ' ' + esc(r.step || "");
    if(r.clean === false) step += ' <span class="warn" title="' + esc(r.planted || "") + '">planted</span>';
    let truth = '<span class="missing">-</span>';
    if(r.truth) truth = r.truth_cite && r.truth_cite.ledger_url ? '<a href="#" data-ledger="' + esc(r.truth_cite.ledger_url) + '">' + esc(r.truth) + '</a>' : esc(r.truth);
    const cells = g.chains.map(ch => {
      const c = r.chains[ch];
      if(!c) return '<span class="missing">-</span>';
      const v = '<span class="v-' + esc(c.verdict) + '">' + esc(c.verdict) + '</span>';
      const link = c.cite && c.cite.ledger_url ? '<a href="#" data-ledger="' + esc(c.cite.ledger_url) + '">' + v + '</a>' : v;
      const mark = c.agree === true ? ' <span class="ok" title="agrees with the truth arm">&#10003;</span>'
                 : c.agree === false ? ' <span class="crit" title="disagrees with the truth arm">&#10007;</span>' : '';
      return link + mark + '<div class="tiny">P_data ' + num(c.p_data, 2) + ' &middot; P_recipe ' + num(c.p_recipe, 2) + '</div>';
    });
    return [step, truth].concat(cells);
  });
  rows.push(['<b>agreement</b>', ''].concat(g.chains.map(ch => { const a = (g.agreement || {})[ch] || {}; return '<b>' + esc(a.agree) + '/' + esc(a.compared) + '</b>'; })));
  return table(head, rows) + '<div class="tiny">Chain verdict against the truth arm (ACCEPT ~ helps, HOLD ~ neutral, REJECT ~ hurts); '
    + 'a cell links to the ledger line of its gate decision. Source: ' + esc(g.source || "-") + '.</div>';
}
function renderExp(d){
  let h = expPicker();
  if(!d.available) return h + unavailable(d);
  const st = d.state || {}, df = d.definition || {};
  const done = st.done != null ? !!st.done : (d.report && d.report.present ? !!d.report.done : null);
  h += '<div class="tiny">record ' + esc(d.file) + ' (' + esc(d.utc) + ', ' + ago(d.age_s) + (d.by ? ', taken by ' + esc(d.by) : '') + ')</div>';
  h += kv([
    ["builder", esc(df.builder) + (df.replay_mode ? ' <span class="tiny">replay ' + esc(df.replay_mode) + '</span>' : '')],
    ["state", esc([d.built === false ? "not built" : "built", done === true ? "done " + (st.done_utc || "") : done === false ? "running" : "no state.json in this snapshot",
                   st.generation != null ? "generation " + st.generation : "", d.abandoned ? "ABANDONED" : ""].filter(x => x).join(", "))],
    ["chains", esc(Object.keys(st.chains || {}).map(k => k + ": " + ((st.chains[k] || {}).phase || "?") + " k=" + ((st.chains[k] || {}).k)).join("; "))],
    d.advance ? ["last advance", d.advance.ok === false ? '<span class="crit">' + esc(d.advance.error_kind || "") + ' ' + esc(short(d.advance.error, 300)) + '</span>'
      : esc(d.advance.skipped ? "skipped: " + d.advance.skipped : "ok, submitted " + ((d.advance.job_ids || []).join(", ") || "nothing"))] : null,
    ["ledger", esc(d.ledger.lines_held) + ' line(s) held through line ' + esc(d.ledger.through_line)
      + (d.ledger.file_lines != null ? ' of ' + esc(d.ledger.file_lines) : '')
      + ' <span class="tiny">' + esc(d.ledger.source || "") + '</span>'
      + (d.ledger.verified === false ? ' <span class="warn">(the copy and the position the ticker recorded disagree)</span>' : '')
      + ' <a href="#" data-ledger="/api/inc/' + encodeURIComponent(d.exp) + '/ledger?tail=20">last lines</a>']
  ]);
  h += '<h4>Steps and chains</h4>' + gridHtml(d.grid);
  const bl = Object.keys(st.blocked || {});
  h += '<h4>Blocked units</h4>' + (bl.length ? table(["unit", "cause", "error"], bl.map(u => [esc(u), esc(st.blocked[u].cause), '<code>' + esc(short(st.blocked[u].error, 300)) + '</code>']))
    + '<div class="tiny">The autopilot lifts a unit blocked by a transient cause once (lever L7); any other block waits for a person on the cluster.</div>'
    : '<div class="ok tiny">none</div>');
  h += '<h4>Jobs</h4><div class="tiny">' + esc(d.jobs_source) + '</div>' + (d.jobs.length ? table(["job", "name", "state", "elapsed"], d.jobs.map(j => [
      esc(j.id), esc(j.name || ""), esc(j.state || ""), esc(j.elapsed || "")])) : '<div class="missing">no job of this experiment is queued</div>')
    + '<div class="tiny">Stopping the runs of an experiment is scancel of the experiment (R3): use Pause and kill in the Campaign card, which goes through the policy gate and is recorded.</div>';
  h += '<div class="ctl"><button data-act="advance">Advance now</button><button data-act="report">Rebuild report</button>'
    + '<button data-act="snapshot">Snapshot now</button><span class="tiny">One cluster call each, as you. A snapshot is kept on the lab; the diagnoses stay those the ticker recorded.</span></div>';
  if((d.ledger.notes || []).length) h += '<details><summary>ledger notes</summary>' + d.ledger.notes.map(n => '<div class="tiny">' + esc(n) + '</div>').join('') + '</details>';
  return h;
}
function diagHtml(x){
  return '<div class="diag">' + pill(x.severity, x.severity) + ' <b>' + esc(x.id) + ' ' + esc(x.name) + '</b>'
    + ((x.levers || []).length ? ' <span class="tiny">levers ' + esc(x.levers.join(", ")) + '</span>' : '')
    + '<div>' + esc(x.summary) + '</div>'
    + citesHtml(x.cites || []) + '</div>';
}
function citesHtml(cs){
  if(!cs.length) return '';
  const li = c => '<li>' + citeHtml(c) + '</li>';
  const head = '<ul class="cites">' + cs.slice(0, 6).map(li).join('') + '</ul>';
  if(cs.length <= 6) return head;
  return head + '<details><summary>' + (cs.length - 6) + ' more cited value(s)</summary><ul class="cites">'
    + cs.slice(6).map(li).join('') + '</ul></details>';
}
function renderDiag(d){
  if(!d.available) return unavailable(d);
  const all = d.diagnoses || [];
  const fired = all.filter(x => x.fired), quiet = all.filter(x => !x.fired);
  let h = d.source === "ticker" ? '<div class="tiny">Recorded by the ticker: ' + esc(d.recorded_in) + '.</div>'
    : '<div class="warn tiny">Preview, not what the ticker recorded: ' + esc(d.note || "") + '.</div>';
  const hf = (d.health && d.health.fired) || [];
  if(hf.length) h += '<h4>Health at the last observation (' + esc(d.health.utc) + ')</h4>' + hf.map(diagHtml).join('');
  h += fired.length ? fired.map(diagHtml).join('') : '<div class="ok">no diagnosis fires</div>';
  h += '<details><summary>' + quiet.length + ' checked and silent, or not evaluable</summary>'
    + quiet.map(x => '<div class="tiny">' + esc(x.id) + ' ' + esc(x.name) + ': ' + esc(x.summary) + '</div>').join('') + '</details>';
  return h;
}
function decideControls(id){
  return '<div class="ctl"><input class="txt" data-why="' + esc(id) + '" placeholder="why (required)">'
    + '<button data-decide="approve" data-id="' + esc(id) + '">Approve</button>'
    + '<button data-decide="deny" data-id="' + esc(id) + '">Deny</button></div>';
}
function itemHtml(it){
  const cls = it.status === "pending" ? "warn" : it.status === "approved" ? "ok" : it.status === "denied" ? "crit" : "info";
  let h = '<div class="item">' + pill(it.status, cls) + ' <b>' + esc(it.lever || it.action) + '</b> <span class="tiny">' + esc(it.action) + ' ' + esc(it.risk) + ' &middot; ' + esc(it.id) + '</span>';
  if((it.argv || []).length) h += '<div><code>' + esc(it.argv.join(" ")) + '</code></div>';
  h += '<div class="tiny">trigger ' + esc((it.trigger || []).join(", ") || "-") + ' &middot; est ' + num(it.est_su, 1) + ' SU &middot; asked by ' + esc(it.requested_by)
    + (it.decided_by ? ' &middot; decided by ' + esc(it.decided_by) + (it.decision_basis ? ' (' + esc(it.decision_basis) + ')' : '') : '') + '</div>';
  if(it.reason) h += '<div class="tiny">' + esc(short(it.reason, 400)) + '</div>';
  (it.brain_notes || []).forEach(n => { h += '<div class="tiny">brain note (' + esc(n.proposed_by) + '): ' + esc(short(n.rationale, 300)) + '</div>'; });
  if(it.status === "pending") h += decideControls(it.id);
  else if(it.status === "approved" && !it.execution) h += '<div class="ctl"><button data-run-approved="' + esc(it.id) + '">Run now</button><span class="tiny">or the ticker runs it on its next tick</span></div>';
  else if(it.execution) h += '<div class="tiny">execution ' + esc(it.execution.phase) + (it.execution.outcome ? ': ' + esc(it.execution.outcome.status) + ' ' + esc((it.execution.outcome.job_ids || []).join(", ")) + ' ' + esc(it.execution.outcome.error || "") : '') + '</div>';
  return h + '<div class="err" data-err="' + esc(it.id) + '"></div></div>';
}
function planHtml(p){
  return '<div class="item tiny"><b>plan ' + esc(p.n) + '</b> ' + esc(p.campaign) + ' &middot; ' + esc(p.exp) + ' &middot; staged ' + esc(p.created_utc || "")
    + ' &middot; ~' + esc(p.tokens_estimated || "?") + ' tokens &middot; <code>' + esc(String(p.sha256 || "").slice(0, 12)) + '</code></div>';
}
function renderProps(d){
  if(!d.available) return unavailable(d);
  const f = d.filed || {};
  const det = f.deterministic || [], br = f.brain || [], oth = f.other || [];
  const fl = d.in_flight;
  let left = '<h4>Deterministic <span class="tiny">round-scheduler:inc-autopilot</span></h4>'
    + (fl ? '<div class="item">' + pill("in flight", "info") + ' <b>' + esc(fl.lever) + '</b> <span class="tiny">' + esc(fl.action) + ' &middot; ' + esc(fl.status) + (fl.approval_id ? ' &middot; approval ' + esc(fl.approval_id) : '') + '</span>'
        + ((fl.argv || []).length ? '<div><code>' + esc(fl.argv.join(" ")) + '</code></div>' : '')
        + '<div class="tiny">trigger ' + esc((fl.trigger || []).join(", ") || "-") + ' &middot; est ' + num(fl.est_gpu_hours, 1) + ' GPU-h</div></div>' : '')
    + (det.length ? det.map(itemHtml).join('') : '<div class="missing">nothing filed</div>');
  const c = d.computed;
  if(c && !c.error){
    left += '<details><summary>What the rules propose on the latest snapshot of ' + esc(d.exp) + ' (' + (c.proposals || []).length + '; the ticker files them, this page does not)</summary>'
      + (c.proposals || []).map(p => '<div class="item"><code>' + esc((p.argv || []).join(" ")) + '</code><div class="tiny">' + esc(p.lever) + ' &middot; trigger ' + esc((p.trigger || []).join(", ")) + ' &middot; est ' + num(p.est_gpu_hours, 1) + ' GPU-h &middot; ' + esc(p.risk) + '</div></div>').join('')
      + (c.operations || []).map(o => '<div class="tiny">operation ' + esc(o.op) + ' (' + esc((o.trigger || []).join(", ")) + '): ' + esc(o.title || "") + '</div>').join('')
      + (c.deferred || []).map(x => '<div class="tiny missing">deferred ' + esc(x.lever) + ' (' + esc(x.trigger) + '): ' + esc(x.reason) + '</div>').join('')
      + (c.refused || []).map(x => '<div class="tiny crit">refused ' + esc(x.lever) + ' (' + esc(x.trigger) + '): ' + esc(x.reason) + '</div>').join('')
      + '</details>';
  } else if(c && c.error){ left += '<div class="err">' + esc(c.error) + '</div>'; }
  let right = '<h4>Brain <span class="tiny">tier2, advisory</span></h4>' + (br.length ? br.map(itemHtml).join('') : '<div class="missing">nothing filed</div>');
  const b = d.brain;
  right += b ? '<div class="tiny">plan ' + esc(b.n) + ' for ' + esc(b.exp) + ': ' + esc(b.status) + (b.counts ? ' (' + esc(b.counts.valid || 0) + ' valid, ' + esc(b.counts.dropped || 0) + ' dropped, ' + esc(b.counts.cards || 0) + ' card(s))' : '') + (b.reason ? ' - ' + esc(short(b.reason, 300)) : '') + '</div>'
    : '<div class="missing tiny">' + (d.brain_enabled ? 'no brain plan yet' : 'the brain is off for this campaign') + '</div>';
  right += (d.plans || []).length ? '<h4>Digests staged</h4>' + d.plans.map(planHtml).join('') : '';
  return '<div class="tiny">' + esc(d.n_pending) + ' waiting on a person. Approve and deny go to the approval queue (/api/brain/' + DOMAIN + '/approvals); the decision is recorded in your name.</div>'
    + '<div class="two"><div>' + left + '</div><div>' + right + '</div></div>'
    + (oth.length ? '<h4>Other</h4>' + oth.map(itemHtml).join('') : '');
}
function field(k, v){ return v ? '<div><span class="tiny">' + esc(k) + ':</span> ' + esc(v) + '</div>' : ''; }
function renderCards(d){
  if(!d.available) return unavailable(d);
  const cs = d.cards || [];
  if(!cs.length) return '<div class="missing">no research card is open</div>';
  return cs.map(c => '<div class="item">' + pill("R4", "warn") + ' <b>' + esc(c.lever || "") + ' ' + esc(c.title || "") + '</b>'
    + '<div class="tiny">' + esc(c.source || "") + ' &middot; ' + esc(c.proposed_by || "") + ' &middot; trigger ' + esc((c.trigger || []).join(", ")) + '</div>'
    + field("Hypothesis", c.hypothesis) + field("Why the menu is not enough", c.why_menu_insufficient)
    + field("Required change", c.required_change) + field("Cheapest test", c.cheapest_test)
    + field("Control", c.control) + field("Success criterion", c.success_criterion)
    + ((c.lit || []).length ? '<div class="tiny">literature: ' + c.lit.map(l => esc(l.paper_id || l)).join(", ") + '</div>' : '')
    + '</div>').join('') + '<div class="tiny">An R4 card is never queued or executed: a person implements it, as a new protocol version.</div>';
}
function pct(v){ return v == null ? '<span class="missing">-</span>' : num(100 * v, 0) + '%'; }
function renderTrack(d){
  if(!d.available) return unavailable(d);
  const lv = Object.keys(d.levers || {}).sort().map(k => { const v = d.levers[k] || {}; return [esc(k), esc(v.scored), esc(v.correct), esc(v.contradicted), esc(v.insufficient), pct(v.accuracy), esc(v.last ? (v.last.child_exp || "") + " " + (v.last.verdict || "") : "")]; });
  const pr = Object.keys(d.proposers || {}).sort().map(k => { const v = d.proposers[k] || {}; return [esc(k), esc(v.plans), esc(v.items), esc(v.valid), esc(v.dropped), pct(v.citation_failure_rate), pct(v.prediction_accuracy), esc(v.scored)]; });
  return '<h4>Per lever</h4>' + table(["lever", "scored", "correct", "contradicted", "insufficient", "accuracy", "last"], lv)
    + '<h4>Per proposer</h4>' + table(["proposer", "plans", "items", "valid", "dropped", "cite failures", "prediction accuracy", "scored"], pr)
    + '<div class="tiny">More autonomy for a brain is earned by this record and granted by a person; it is never self-granted.</div>';
}
function renderReplay(d){
  if(!d.available) return unavailable(d);
  const r = d.replay || {}, rec = r.recorded || {};
  let h = '<div>' + (r.passed ? pill("passed", "ok") : pill("not passed", "warn")) + ' <span class="tiny">' + esc(r.reason) + '</span></div>';
  if(rec.status) h += '<div class="tiny">recorded ' + esc(rec.recorded_utc) + ' for code ' + esc(String(rec.code_hash || "").slice(0, 12)) + ' (now ' + esc(String(r.code_hash_now || "").slice(0, 12)) + ')</div>'
    + table(["case", "result"], Object.keys(rec.cases || {}).sort().map(k => { const v = rec.cases[k]; return [esc(k), v === "pass" ? '<span class="ok">pass</span>' : v === "skip" ? '<span class="warn">skip</span>' : '<span class="crit">' + esc(v) + '</span>']; }));
  const ps = d.prospective || [];
  h += '<h4>Prospective (R4b)</h4>' + (ps.length ? table(["exp", "rules", "outcome", "ready", "replay", "recipes", "decided", "against the real loop"], ps.map(p => [
      esc(p.exp), esc(p.rules_version || "unversioned") + (p.current_rules ? ' <span class="ok">current</span>' : ' <span class="tiny">history</span>'),
      esc(p.outcome), esc(p.ready), esc(p.replay_mode), esc((p.recipes || []).join(",")), esc(p.decided_utc),
      p.superseded ? '<span class="tiny">superseded: not compared (only the record of the current rules is)</span>'
        : p.compare ? (p.compare.match ? '<span class="ok">match</span>' : '<span class="crit">differs</span>') + ' <span class="tiny">' + esc(p.compare.realloop) + '</span>' : '<span class="missing">no real loop yet</span>'])) : '<div class="missing">no prospective record yet</div>');
  return h + '<div class="tiny">' + esc(d.note) + '</div>';
}
function renderFinal(d){
  if(!d || !d.available) return unavailable(d);
  const f = d.display_only;
  if(!f) return '<div class="missing">the latest snapshot of ' + esc(d.exp) + ' carries no report final table</div>';
  const exams = [];
  (f.rows || []).forEach(r => Object.keys((r && r.exams) || {}).forEach(e => { if(exams.indexOf(e) < 0) exams.push(e); }));
  const rows = (f.rows || []).map(r => [esc(r.model), esc(r.runs)].concat(exams.map(e => {
    const x = (((r.exams || {})[e]) || {}).twelve || {};
    return x.mean == null ? '-' : num(x.mean, 4) + (x.sd != null ? ' &plusmn; ' + num(x.sd, 4) : '');
  })));
  return '<div class="disp"><b>' + esc(f.label).toUpperCase() + '.</b> Every exam of the report is shown to people here; '
    + 'no decision reads anything but the dev column, and it reads that from the decision part of the snapshot, not from this table.</div>'
    + table(["model", "runs"].concat(exams.map(e => e + " 12-class mAP50-95")), rows);
}
function openLedger(url){
  const card = document.getElementById("c_ledger");
  card.style.display = "";
  set("b_ledger", "loading...");
  getJSON(url).then(d => {
    if(!d.available){ set("b_ledger", unavailable(d) + '<button data-close-ledger="1">Close</button>'); return; }
    let h = '<div class="tiny">' + esc(d.exp) + ' ledger, lines ' + esc(d.first) + ' to ' + esc(d.last) + ', held on the lab through line ' + esc(d.held_through)
      + (d.verified === false ? ' <span class="warn">(the held copy is not verified against the cluster hash)</span>' : '') + '</div>';
    if(d.reason) h += '<div class="warn">' + esc(d.reason) + '</div>';
    if((d.missing || []).length) h += '<div class="warn tiny">lines not held on the lab: ' + esc(d.missing.join(", ")) + '</div>';
    h += (d.rows || []).map(r => '<div class="' + (r.line === d.line ? 'hl' : '') + '"><b>line ' + esc(r.line) + '</b> <code>' + esc((r.entry && r.entry.id) || "") + '</code>'
      + '<pre class="led">' + esc(JSON.stringify(r.entry, null, 1)) + '</pre></div>').join('');
    set("b_ledger", h + '<button data-close-ledger="1">Close</button>');
    card.scrollIntoView({behavior: "smooth", block: "start"});
  }).catch(e => set("b_ledger", fail(e)));
}
function campAct(what){
  const body = {campaign: S.campaign};
  if(what === "enable") body.enabled = true;
  else if(what === "disable") body.enabled = false;
  else if(what === "resume") body.resume = true;
  else if(what === "pause"){ const w = val("in_pause"); if(!w){ say("a pause needs a reason", true); return; } body.pause = w; }
  else if(what === "goal"){
    const k = val("in_goal_kind");
    if(k === "none") body.goal = null;
    else if(k === "diagnosis"){
      const t = val("in_goal_diag"), i = t.indexOf(":");
      if(i < 1){ say("a diagnosis goal is ID:NAME, e.g. D4:decision_slot_ready", true); return; }
      body.goal = {kind: "diagnosis", id: t.slice(0, i), name: t.slice(i + 1)};
    } else {
      const e = val("in_goal_exp");
      if(!e){ say("an experiment goal needs the experiment's name", true); return; }
      body.goal = {kind: "exp_done", exp: e};
    }
  }
  else if(what === "budget"){
    const e = val("in_env"), dc = val("in_daily");
    if(e !== "") body.envelope_su = Number(e);
    if(dc !== "") body.daily_cap_su = Number(dc);
  }
  else if(what === "autonomy-on"){
    if(!confirm("Let the platform run R3 builds of L1, L2, L5, L8 and L9 inside the envelope without asking, authorised in your name?")) return;
    body.autonomy = "envelope";
  }
  else if(what === "autonomy-off") body.autonomy = "off";
  else if(what === "kill"){ kill(); return; }
  postJSON("/api/inc/campaign", body).then(d => {
    if(d.ok){ say("saved" + (d.warning ? " - " + d.warning : ""), !!d.warning); loadAll(); }
    else say(d.reason || ("refused (" + d._status + ")"), true);
  }).catch(e => say(String(e), true));
}
function kill(){
  const exp = S.current, why = val("in_kill");
  if(!exp){ say("the campaign has no current experiment", true); return; }
  if(!why){ say("killing an experiment needs a reason", true); return; }
  if(!confirm("Pause the campaign and cancel every job of " + exp + "? The experiment is marked abandoned on the cluster; a person removes the marker to resume it.")) return;
  postJSON("/api/inc/campaign", {campaign: S.campaign, pause: "kill " + exp + ": " + why}).then(d1 => {
    if(!d1.ok){ say("pause refused: " + (d1.reason || d1._status), true); return; }
    return postJSON("/api/inc/action/cancel", {campaign: S.campaign, params: {exp: exp}, reason: why}).then(d2 => {
      say(d2.ok ? "cancelled " + exp : "cancel " + (d2.status || d2._status) + ": " + (d2.reasons || [d2.reason]).join("; "), !d2.ok);
      loadAll();
    });
  }).catch(e => say(String(e), true));
}
function expAct(verb){
  if(!S.exp){ say("no experiment selected", true); return; }
  say(verb + " " + S.exp + " ...", false);
  postJSON("/api/inc/action/" + verb, {campaign: S.campaign, params: {exp: S.exp}}).then(d => {
    say(verb + ": " + (d.status || d._status) + (d.kept_as ? " (kept as " + d.kept_as + ")" : "") + (d.reasons && d.reasons.length ? " - " + d.reasons.join("; ") : d.reason ? " - " + d.reason : ""), !d.ok);
    if(d.ok) loadAll();
  }).catch(e => say(String(e), true));
}
function decide(id, what){
  const box = document.querySelector('[data-why="' + id + '"]');
  const err = document.querySelector('[data-err="' + id + '"]');
  const why = ((box && box.value) || "").trim();
  if(!why){ if(err) err.textContent = "a decision needs a reason"; return; }
  if(err) err.textContent = "sending...";
  postJSON("/api/brain/" + DOMAIN + "/approvals/" + encodeURIComponent(id), {decision: what, reason: why}).then(d => {
    if(d.ok){ say(what + "d " + id, false); loadAll(); } else if(err) err.textContent = d.reason || "refused";
  }).catch(e => { if(err) err.textContent = String(e); });
}
function runApproved(id, ack){
  if(!ack && !confirm("Run approved item " + id + " now, in your name?")) return;
  const body = {campaign: S.campaign, approval_id: id};
  if(ack) body.acknowledge_non_prospective = ack;
  postJSON("/api/inc/action/execute-approved", body).then(d => {
    if(d._status === 409 && !ack){
      // A real loop that D4's frozen decision does not cover: only a person's stated reason overrides it.
      const why = prompt((d.reasons || []).join("; ") + " -- to run it anyway, give your reason (the execution log records it):");
      if(why && why.trim()) return runApproved(id, why.trim());
    }
    say("run " + id + ": " + (d.status || d._status) + (d.reasons && d.reasons.length ? " - " + d.reasons.join("; ") : ""), !d.ok);
    loadAll();
  }).catch(e => say(String(e), true));
}
function create(){
  const name = val("in_cname"), exp = val("in_cexp");
  if(!name || !exp){ say("a campaign needs a name and the experiment it follows first", true); return; }
  postJSON("/api/inc/campaign", {campaign: name, create: true, exp: exp}).then(d => {
    if(d.ok){ S.campaign = name; say("created " + name + " (disabled)", false); loadAll(); } else say(d.reason || ("refused (" + d._status + ")"), true);
  }).catch(e => say(String(e), true));
}
document.addEventListener("click", function(ev){
  const t = ev.target.closest ? ev.target.closest("[data-ledger],[data-exp],[data-decide],[data-run-approved],[data-camp],[data-act],[data-close-ledger],[data-create],[data-refresh]") : null;
  if(!t) return;
  ev.preventDefault();
  if(t.hasAttribute("data-ledger")) openLedger(t.getAttribute("data-ledger"));
  else if(t.hasAttribute("data-exp")){ S.exp = t.getAttribute("data-exp"); loadRest(); }
  else if(t.hasAttribute("data-decide")) decide(t.getAttribute("data-id"), t.getAttribute("data-decide"));
  else if(t.hasAttribute("data-run-approved")) runApproved(t.getAttribute("data-run-approved"));
  else if(t.hasAttribute("data-camp")) campAct(t.getAttribute("data-camp"));
  else if(t.hasAttribute("data-act")) expAct(t.getAttribute("data-act"));
  else if(t.hasAttribute("data-close-ledger")) document.getElementById("c_ledger").style.display = "none";
  else if(t.hasAttribute("data-create")) create();
  else if(t.hasAttribute("data-refresh")) loadAll();
});
document.addEventListener("change", function(ev){
  const t = ev.target;
  if(!t || !t.getAttribute) return;
  const which = t.getAttribute("data-sel");
  if(which === "campaign"){ S.campaign = t.value; S.exp = ""; loadAll(); }
  else if(which === "exp"){ S.exp = t.value; loadRest(); }
});
function load(id, url, fn){ return getJSON(url).then(d => set(id, fn(d))).catch(e => set(id, fail(e))); }
function loadRest(){
  load("b_lineage", "/api/inc/lineage" + q(), lineageHtml);
  if(S.exp){
    const base = "/api/inc/" + encodeURIComponent(S.exp);
    load("b_exp", base + "/snapshot" + q(), d => { set("b_final", renderFinal(d)); return renderExp(d); });
    load("b_diag", base + "/diagnoses" + q(), renderDiag);
  } else {
    set("b_exp", '<div class="missing">no experiment is known on the lab yet</div>');
    set("b_diag", '<div class="missing">no experiment to diagnose</div>');
    set("b_final", '<div class="missing">no experiment</div>');
  }
  load("b_props", "/api/inc/proposals" + q({exp: S.exp}), d => { set("b_cards", renderCards(d)); return renderProps(d); });
  load("b_track", "/api/inc/track", renderTrack);
  load("b_replay", "/api/inc/replay", renderReplay);
}
function loadAll(){
  getJSON("/api/inc/campaign").then(d => {
    const cs = d.campaigns || [];
    const c = cs.find(x => x.name === S.campaign) || cs[0] || null;
    S.campaign = c ? c.name : "";
    S.current = c ? c.exp : null;
    S.known = (d.known_exps || []).slice();
    ((c && c.exps) || []).forEach(e => { if(S.known.indexOf(e) < 0) S.known.push(e); });
    if(!S.exp) S.exp = S.current || S.known[S.known.length - 1] || "";
    set("b_campaign", renderCampaign(d, c));
    loadRest();
  }).catch(e => { set("b_campaign", fail(e)); loadRest(); });
}
loadAll();
setInterval(function(){
  const a = document.activeElement;
  const busy = a && (a.tagName === "INPUT" || a.tagName === "SELECT" || a.tagName === "TEXTAREA");
  if(!busy && !document.hidden) loadAll();
}, 120000);
</script></body></html>"""
